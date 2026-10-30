"""Generate one registry-invoke ACLNN submission containing the selected Micro ops."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import yaml

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
PATHS = (
    "level1/swi_glu", "level2/cummin", "level2/gather", "level2/rms_norm",
    "level2/scatter", "level2/softmax", "level2/image_resize_normalize",
    "level3/add_rms_norm_dynamic_quant", "level3/quant_matmul", "level3/top_k",
    "level3/transpose", "level3/grouped_matmul", "level4/mla",
    "level4/linear_cross_entropy", "level5/kimi_delta_attention",
)
BY_NAME = {Path(path).name: path for path in PATHS}
GE_TYPES = {"float16": "FLOAT16", "bfloat16": "BF16", "float32": "FLOAT",
            "int8": "INT8", "int16": "INT16", "int32": "INT32", "int64": "INT64",
            "uint8": "UINT8", "bool": "BOOL"}
ATTR_TYPES = {"int": "Int", "float": "Float", "bool": "Bool", "str": "String", "List[int]": "ListInt"}
CPP_TYPES = {"Tensor": "const at::Tensor&", "List[Tensor]": "at::TensorList",
             "int": "int64_t", "float": "double", "bool": "bool", "str": "const std::string&",
             "List[int]": "at::IntArrayRef", "Optional[Tensor]": "const c10::optional<at::Tensor>&",
             "Optional[str]": "const c10::optional<std::string>&"}


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Official scaffold changed; review adaptation: {before}")
    return source.replace(before, after, 1)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def parse_schema(source):
    # Reuse the parser that will consume the generated torch.library declaration.
    from torch._C import parse_schema as torch_parse_schema

    schema = torch_parse_schema(source)
    if not schema.name.isidentifier() or schema.overload_name:
        raise ValueError(f"Expected a plain operator name: {schema.name}")
    return schema


def load_spec(source):
    spec = yaml.safe_load(source.read_text(encoding="utf-8"))["operator"]
    schema = parse_schema(spec["schema"])
    arguments = {argument.name: argument for argument in schema.arguments}
    declared = [field["name"] for key in ("inputs", "attrs") for field in spec.get(key, [])]
    if len(declared) != len(set(declared)) or set(arguments) != set(declared):
        raise ValueError(f"proto fields and schema arguments differ: {source}")
    if [field["name"] for field in spec["outputs"]] != [value.name for value in schema.returns]:
        raise ValueError(f"proto outputs and schema returns differ: {source}")
    if set(arguments) & {value.name for value in schema.returns}:
        raise ValueError("In-place input/output aliases need an explicit implementation")
    for argument in schema.arguments:
        if str(argument.type) not in CPP_TYPES:
            raise ValueError(f"Unsupported schema type: {argument.name}: {argument.type}")
    for key in ("inputs", "outputs", "attrs"):
        fields = spec.get(key, [])
        for field, argument in zip(fields, schema.returns if key == "outputs" else
                                   [arguments[field["name"]] for field in fields]):
            optional = argument.type.kind() == "OptionalType"
            value_type = argument.type.getElementType() if optional else argument.type
            field.update(optional=optional, schema_type=str(value_type),
                         dynamic=str(value_type) == "List[Tensor]",
                         has_default=argument.has_default_value(), default=argument.default_value)
            if key == "attrs" and (str(value_type) not in ATTR_TYPES or optional and str(value_type) != "str"):
                raise ValueError(f"Unsupported attribute type: {field['name']}: {argument.type}")
            if key != "attrs" and str(value_type) not in ("Tensor", "List[Tensor]"):
                raise ValueError(f"Expected Tensor or Tensor[]: {field['name']}")
            if key == "outputs" and optional:
                raise ValueError("Optional outputs need an explicit implementation")
            for dtype in field.get("dtype", []):
                if dtype not in GE_TYPES:
                    raise ValueError(f"Unsupported dtype: {dtype}")
    # Keep the prototype's order within inputs/attrs: CANN uses inputs, attrs, outputs.
    name = schema.name
    return dict(spec, id=name, custom="".join(part.capitalize() for part in name.split("_")) + "Custom")


def spec_for(repo, name):
    spec = load_spec(repo / "tasks" / BY_NAME[name] / "proto.yaml")
    if spec["id"] != name:
        raise ValueError(f"Task directory and schema name differ: {name}")
    return dict(spec, task_dir="tasks/" + BY_NAME[name])


def scaffold(repo, output):
    example = repo / "examples/aclnn_launch_example"
    # Copy upstream infrastructure, not Add/Sqrt implementations or documentation.
    for relative in ("CMakeLists.txt", "setup.py", "requirements.txt", "csrc/extension.cpp",
                     "csrc/ops/CMakeLists.txt", "csrc/ops/_common/aclnn_common.h"):
        write(output / relative, (example / relative).read_text(encoding="utf-8"))
    for directory in ("cmake", "scripts"):
        for source in (example / directory).iterdir():
            if source.is_file():
                write(output / directory / source.name, source.read_text(encoding="utf-8"))
    write(output / "csrc/ops/CMakeLists.txt", (TEMPLATES / "ops.CMakeLists.txt").read_text(encoding="utf-8"))
    macros = output / "cmake/func.cmake"
    link_sources = '''    # Kernel links are derived build inputs, never the editable source of truth.
    file(GLOB _KERNEL_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${TILING_INCLUDE_DIR}/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${TILING_INCLUDE_DIR}/*.cpp")
    foreach(_KERNEL_SOURCE IN LISTS _KERNEL_SOURCES)
        get_filename_component(_KERNEL_NAME "${_KERNEL_SOURCE}" NAME)
        set(_KERNEL_LINK "${PROJECT_SOURCE_DIR}/csrc/ops/op_kernel/${_KERNEL_NAME}")
        if(EXISTS "${_KERNEL_LINK}")
            message(FATAL_ERROR "Duplicate kernel filename: ${_KERNEL_NAME}")
        endif()
        file(CREATE_LINK "${_KERNEL_SOURCE}" "${_KERNEL_LINK}" RESULT _LINK_RESULT)
        if(NOT _LINK_RESULT STREQUAL "0")
            message(FATAL_ERROR "Cannot create kernel hard link: ${_LINK_RESULT}")
        endif()
    endforeach()

'''
    write(macros, replace_once(macros.read_text(encoding="utf-8"),
          '    message(STATUS "Registering ACLNN op: ${OP_TYPE}")',
          link_sources + '    message(STATUS "Registering ACLNN op: ${OP_TYPE}")'))
    setup = output / "setup.py"
    write(setup, replace_once(setup.read_text(encoding="utf-8"), "'-DBUILD_PYTHON_EXT=ON'",
                             "'-DASCEND_COMPUTE_UNIT=ascend910_93', '-DBUILD_PYTHON_EXT=ON'"))
    # The sample warns and continues after compiler errors; submissions must fail.
    write(output / "scripts/build_run.sh", (TEMPLATES / "build_run.sh").read_text(encoding="utf-8"))
    write(output / "scripts/build_wheel.sh", (TEMPLATES / "build_wheel.sh").read_text(encoding="utf-8"))
    write(output / "build.sh", (TEMPLATES / "build.sh").read_text(encoding="utf-8"))
    original_init = (example / "cann_bench/__init__.py").read_text(encoding="utf-8")
    write(output / "cann_bench/__init__.py", original_init.split("# Direct function calls:", 1)[0])
    write(output / ".gitignore", "/build/\n/build_py/\n/dist/\n/csrc/ops/op_kernel/\n/*.egg-info/\n__pycache__/\n*.pyc\n/cann_bench/*.so\n/_compile.log\n")
    if (repo / "LICENSE").is_file():
        shutil.copyfile(repo / "LICENSE", output / "LICENSE")


def declaration(spec):
    lines = [f"class {spec['custom']} : public OpDef {{", "public:",
             f"    explicit {spec['custom']}(const char* name) : OpDef(name) {{"]
    for key, method in (("inputs", "Input"), ("outputs", "Output")):
        for field in spec[key]:
            kind = "DYNAMIC" if field["dynamic"] else "OPTIONAL" if field["optional"] else "REQUIRED"
            types = ", ".join("ge::DT_" + GE_TYPES[t] for t in field["dtype"])
            lines += [f'        this->{method}("{field["name"]}").ParamType({kind})',
                      f"            .DataTypeList({{{types}}}).FormatList({{ge::FORMAT_ND}});"]
    for field in spec.get("attrs", []):
        method = ATTR_TYPES[field["schema_type"]]
        default = field["default"]
        # CANN String attrs are not nullable. The plugin encodes None as an empty string.
        if field["optional"] and default is None:
            default = ""
        kind = "OPTIONAL" if field["has_default"] else "REQUIRED"
        value = json.dumps(default) if default is not None else ""
        if isinstance(default, list):
            value = value.replace("[", "{").replace("]", "}")
        lines += [f'        this->Attr("{field["name"]}").AttrType({kind}).{method}({value});']
    lines += ['        this->AICore().AddConfig("ascend910_93");', "    }", "};",
              f"OP_ADD({spec['custom']});"]
    return "\n".join(lines)


def host(spec):
    name = spec["id"]
    infer = f'''#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

namespace {spec['custom']}Impl {{
static ge::graphStatus InferShape(gert::InferShapeContext* c) {{
    // TODO: infer output shapes (including the number of dynamic outputs).
    return ge::GRAPH_FAILED;
}}
static ge::graphStatus InferType(gert::InferDataTypeContext* c) {{
    // TODO: infer output dtypes from the operator's contract.
    return ge::GRAPH_FAILED;
}}
IMPL_OP_INFERSHAPE({spec['custom']}).InferShape(InferShape).InferDataType(InferType);
}}
'''
    tiling = f'''#include "register/op_def_registry.h"
#include "../../op_kernel/arch22/{name}_tiling_data.h"
#include "../../op_kernel/arch22/{name}_tiling_key.h"

namespace {spec['custom']}Impl {{
static ge::graphStatus Tile(gert::TilingContext* c) {{
    // TODO: serialize Tiling, then set blockDim, tilingKey and workspace sizes.
    return ge::GRAPH_FAILED;
}}
IMPL_OP_OPTILING({spec['custom']}).Tiling(Tile);
}}
'''
    definition = '#include "register/op_def_registry.h"\n\nnamespace ops {\n' + declaration(spec) + '\n}\n'
    return {f"{name}_def.cpp": definition, f"{name}_infershape.cpp": infer,
            f"arch22/{name}_tiling.cpp": tiling}


def kernel(spec):
    names = [field["name"] for field in spec["inputs"] + spec["outputs"]]
    args = ", ".join("GM_ADDR " + name for name in [*names, "workspace", "tiling"])
    namespace = spec['custom'] + "Kernel"
    call_names = ", ".join(names)
    header = f'''#pragma once
#include "kernel_operator.h"
#include "{spec['id']}_tiling_data.h"
#include "{spec['id']}_tiling_key.h"

namespace {namespace} {{
// TODO: implement {spec['name']} on the NPU. No reference or placeholder values are computed.
__aicore__ inline void Compute({', '.join('GM_ADDR ' + name for name in names)}, const Tiling& data) {{
}}
}}
'''
    entry = f'''#include "{spec['id']}.h"
using {namespace}::Tiling;

extern "C" __global__ __aicore__ void {spec['id']}_custom({args}) {{
    REGISTER_TILING_DEFAULT(Tiling);
    GET_TILING_DATA_WITH_STRUCT(Tiling, data, tiling);
    if (TILING_KEY_IS({namespace}::kTilingKey)) {{
        {namespace}::Compute({call_names}, data);
    }}
}}
'''
    tiling = f'''#pragma once
#include <cstdint>
namespace {namespace} {{
struct Tiling {{
    uint64_t reserved; // TODO: replace with this operator's tiling parameters.
}};
}}
'''
    key = f"#pragma once\n#include <cstdint>\nnamespace {namespace} {{ constexpr uint64_t kTilingKey = 0; }}\n"
    return {f"{spec['id']}.h": header, f"{spec['id']}_kernel.cpp": entry,
            f"{spec['id']}_tiling_data.h": tiling, f"{spec['id']}_tiling_key.h": key}


def plugin(spec):
    schema = parse_schema(spec["schema"])
    arguments = ", ".join(CPP_TYPES[str(arg.type)] + " " + arg.name for arg in schema.arguments)
    forwarded = ", ".join(arg.name for arg in schema.arguments)
    output_types = ["std::vector<at::Tensor>" if field["dynamic"] else "at::Tensor" for field in spec["outputs"]]
    result_type = output_types[0] if len(output_types) == 1 else "std::tuple<" + ", ".join(output_types) + ">"
    conversions, values = [], []
    for field in spec["inputs"] + spec.get("attrs", []):
        name = field["name"]
        if field["schema_type"] == "str":
            value = f'{name}.value_or("")' if field["optional"] else name
            conversions += [f"    std::string ascopt_{name}_text = {value};",
                            f"    const char* ascopt_{name} = ascopt_{name}_text.c_str();"]
            name = "ascopt_" + name
        values.append(name)
    for index, field in enumerate(spec["outputs"]):
        name = field["name"]
        value = "outputs" if len(output_types) == 1 else f"std::get<{index}>(outputs)"
        cpp = "at::TensorList " if field["dynamic"] else "auto& "
        conversions.append(f"    {cpp}{name} = {value};")
        values.append(name)
    return f'''#include <torch/library.h>
#include <tuple>
#include <vector>
#include "../../_common/aclnn_common.h"

namespace {{
{result_type} infer_output({arguments}) {{
    // TODO: allocate outputs with the correct shape/dtype/device. No tensor computation here.
    TORCH_CHECK(false, "{spec['id']}: output allocation is not implemented");
    return {{}};
}}

{result_type} invoke({arguments}) {{
    auto outputs = infer_output({forwarded});
{chr(10).join(conversions)}
    ACLNN_CMD(aclnn{spec['custom']}, {', '.join(values)});
    return outputs;
}}
}}
TORCH_LIBRARY_FRAGMENT(cann_bench, m) {{
    m.def({json.dumps(spec['schema'])});
}}
TORCH_LIBRARY_IMPL(cann_bench, Meta, m) {{
    m.impl("{spec['id']}", infer_output);
}}
TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m) {{
    m.impl("{spec['id']}", invoke);
}}
'''


def emit_operator(project, spec):
    directory = project / "csrc/ops" / spec["id"]
    for folder, files in (("op_host", host(spec)), ("op_kernel/arch22", kernel(spec))):
        for filename, content in files.items():
            write(directory / folder / filename, content)
    write(directory / "op_plugin" / (spec["id"] + "_plugin.cpp"), plugin(spec))
    write(directory / "CMakeLists.txt", f'''set(HOST_SOURCES
    ${{CMAKE_CURRENT_SOURCE_DIR}}/op_host/{spec['id']}_def.cpp
    ${{CMAKE_CURRENT_SOURCE_DIR}}/op_host/{spec['id']}_infershape.cpp
    ${{CMAKE_CURRENT_SOURCE_DIR}}/op_host/arch22/{spec['id']}_tiling.cpp)
# Empty API sources select the official npu_op_code_gen ACLNN API generator.
register_aclnn_op({spec['custom']} "${{HOST_SOURCES}}" ""
    op_kernel {spec['id']}_kernel.cpp op_kernel/arch22 .)
register_aclnn_plugin("${{CMAKE_CURRENT_SOURCE_DIR}}/op_plugin/{spec['id']}_plugin.cpp" .)
''')


def python_binding(source):
    schema = parse_schema(source)
    parameters, values = [], []
    keyword_only = False
    for argument in schema.arguments:
        if argument.kwarg_only and not keyword_only:
            parameters.append("*")
            keyword_only = True
        default = "=" + repr(argument.default_value) if argument.has_default_value() else ""
        parameters.append(argument.name + default)
        values.append(f"{argument.name}={argument.name}" if argument.kwarg_only else argument.name)
    return (f"def {schema.name}({', '.join(parameters)}):\n"
            f"    return torch.ops.cann_bench.{schema.name}({', '.join(values)})\n")


def public_exports(project, schemas):
    marker = "# Generated operator exports.\n"
    init = project / "cann_bench/__init__.py"
    preamble = init.read_text(encoding="utf-8").split(marker, 1)[0]
    write(init, preamble + marker + "\n".join(f"from .operators import {name}" for name in schemas) + "\n")
    write(project / "cann_bench/operators.py", '"""Generated from official operator schemas."""\n\nimport torch\n\n\n'
          + "\n\n".join(python_binding(source) for source in schemas.values()))
    write(project / "submission.json", json.dumps({"tasks_version": "2.0.0", "soc": "ascend910_93",
          "operators": list(schemas), "schemas": schemas}, indent=2) + "\n")


def generate(repo, output, names):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".generate-", dir=output.parent) as temporary:
        project = Path(temporary) / "submission"
        scaffold(repo, project)
        schemas = {}
        for name in names:
            spec = spec_for(repo, name)
            emit_operator(project, spec)
            schemas[name] = spec["schema"]
        public_exports(project, schemas)
        project.rename(output)
    print(f"Generated {output} with {len(names)} operators")


def add_operator(project, proto):
    spec = load_spec(proto)
    metadata = json.loads((project / "submission.json").read_text(encoding="utf-8"))
    target = project / "csrc/ops" / spec["id"]
    if target.exists() or spec["id"] in metadata["operators"]:
        raise FileExistsError(f"Refusing to overwrite operator {spec['id']}")
    with tempfile.TemporaryDirectory(prefix=".add-", dir=project.parent) as temporary:
        staging = Path(temporary)
        emit_operator(staging, spec)
        (staging / "csrc/ops" / spec["id"]).rename(target)
    public_exports(project, {**metadata["schemas"], spec["id"]: spec["schema"]})
    print(f"Added {spec['id']} to {project}")


EXTENSIONS = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".yaml", ".yml", ".md", ".txt", ".json",
              ".cmake", ".sh", ".bash", ".ini", ".proto", ".py", ".temp", ".info", ".1", ".asc", ".so", ".o", ".whl"}
EXCLUDE = {"build", "build_py", "build_out", "dist", "__pycache__", ".git", ".ascopt"}


def source_files(root):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDE or part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.parts[:3] == ("csrc", "ops", "op_kernel"):
            continue
        if path.is_symlink():
            raise ValueError(f"Symbolic links cannot be submitted: {relative}")
        if not path.is_file() or path.name == "_compile.log" or path.suffix in {".pyc", ".pyo", ".so", ".o"}:
            continue
        if path.suffix and path.suffix not in EXTENSIONS:
            raise ValueError(f"Unsupported source extension: {relative}")
        yield relative, path


def export_zip(root, names, destination):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="micro-export-") as temporary:
        output = Path(temporary) / "submission"
        metadata = json.loads((root / "submission.json").read_text(encoding="utf-8"))
        if (metadata["tasks_version"] != "2.0.0" or metadata["soc"] != "ascend910_93"
                or not set(names).issubset(metadata["operators"])):
            raise ValueError(f"Operator selection is not present in {root}")
        for relative, source in source_files(root):
            if relative.as_posix() == "submission.json":
                continue
            if (len(relative.parts) > 3 and relative.parts[:2] == ("csrc", "ops")
                    and relative.parts[2] != "_common" and relative.parts[2] not in names):
                continue
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        public_exports(output, {name: metadata["schemas"][name] for name in names})
        files = list(source_files(output))
        entries = {rel.parent for rel, _ in files}
        if len(files) + len(entries) > 5000 or sum(p.stat().st_size for _, p in files) > 500 * 1024**2:
            raise ValueError("ZIP would exceed the website's entry or unpacked-size limit")
        archive = Path(temporary) / "submission.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as stream:
            for relative, source in files:
                if len(relative.parts) > 16 or any(len(part) > 255 for part in relative.parts):
                    raise ValueError(f"ZIP path exceeds website limits: {relative}")
                stream.write(source, relative.as_posix())
        if archive.stat().st_size > 50 * 1024**2:
            raise ValueError("Compressed ZIP exceeds 50 MiB")
        shutil.copyfile(archive, destination)
    print(f"Exported {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "add", "export"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--submission", type=Path, help="Default: <repo>/submission")
    parser.add_argument("--operators", nargs="+")
    parser.add_argument("--proto", type=Path, help="Official proto.yaml for add")
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = (args.submission or repo / "submission").resolve()
    if (repo / "tasks/metadata/VERSION").read_text(encoding="utf-8").splitlines()[0] != "2.0.0":
        parser.error("This generator targets Micro-official-Tasks 2.0.0")
    if args.action == "add":
        if not args.proto or args.operators:
            parser.error("add requires --proto and does not use --operators")
        add_operator(root, args.proto.resolve())
        return
    names = args.operators or (list(BY_NAME) if args.action == "generate" else
                              json.loads((root / "submission.json").read_text(encoding="utf-8"))["operators"])
    if len(set(names)) != len(names):
        parser.error("Operator selection contains duplicates")
    if args.action == "generate":
        if not set(names).issubset(BY_NAME):
            parser.error("For an additional operator, use add --proto <proto.yaml>")
        generate(repo, root, names)
    else:
        if not args.zip:
            parser.error("export requires --zip")
        export_zip(root, names, args.zip.resolve())


if __name__ == "__main__":
    main()

"""
project_agent.py - turns a lab question into a zipped NetBeans (Ant) project, and checks it first.

    plan -> generate -> compile -> run -> review -> fix (loop) -> package

Needs one function from the caller:   call_ai(prompt, system, role) -> str
(role: planner / coder / fixer / reviewer) and a JDK (javac + java) on PATH.

Packaging: by default the NetBeans files (nbproject/project.xml, project.properties, build.xml,
manifest.mf) are generated. If a folder named nb_template/ (override: NB_TEMPLATE_DIR) exists next to
this file and contains an nbproject/ folder copied from an empty "Java Application" project made in
YOUR NetBeans, that is used instead - the safest option, because it carries the real build-impl.xml.
"""
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

JAVA_RELEASE = os.getenv("JAVA_RELEASE", "17")  # match the JDK in your lab / NetBeans
NB_TEMPLATE_DIR = os.getenv("NB_TEMPLATE_DIR", "nb_template")
MAX_FIX_ROUNDS = 5
MAX_REVIEWS = 2  # review is the least reliable check, so cap it

# ---------------------------------------------------------------- prompts
PROJECT_SYSTEM_PROMPT = (
    "You are a college student creating a Java project for a NetBeans IDE assignment. "
    "Generate ALL the Java source files needed for a complete, working project. "
    "Write code like a beginner — simple variable names, basic loops, straightforward logic. "
    "Always include all necessary import statements. "
    "For GUI programs, use javax.swing and java.awt. "
    "For JDBC programs, use java.sql.* with DriverManager, Connection, PreparedStatement, ResultSet. "
    "For Spring programs, include proper annotations (@SpringBootApplication, @RestController, etc.). "
    "\n\n"
    "CRITICAL OUTPUT FORMAT RULES:\n"
    "1. Output ONLY Java source files, nothing else.\n"
    "2. For EACH file, use this EXACT format:\n"
    "===FILE:packagename/ClassName.java===\n"
    "package packagename;\n\n"
    "import ...;\n\n"
    "public class ClassName {\n"
    "    // code here\n"
    "}\n"
    "===END_FILE===\n\n"
    "3. The path after FILE: must be the package directory + filename (e.g. myapp/Main.java).\n"
    "4. If no package is needed, just use the filename (e.g. Main.java).\n"
    "5. Every class MUST have its own file.\n"
    "6. One file MUST contain a public static void main(String[] args) method.\n"
    "7. Do NOT include any explanation text outside the file blocks.\n"
    "8. Do NOT use markdown code fences (```).\n"
)

PLAN_SYSTEM = "You are a Java instructor planning how to test a small lab project. Reply with JSON only."
PLAN_PROMPT = """Lab question:
{question}

Return JSON only, in exactly this shape:
{{"requirements": ["one item per thing the question asks for"],
  "type": "console",
  "test_inputs": ["stdin text for one test run"],
  "needs_libraries": [],
  "needs_external_services": false}}

Rules:
- "type" is "console" or "gui".
- test_inputs: realistic stdin for 1-3 test runs of a console program. Each run must make the program finish (include the exit option if it shows a menu). Use [] if the program reads no input, or is a gui.
- needs_libraries: third-party jars needed to COMPILE it (for example Spring or Gson), else []. A JDBC driver is only needed at run time, so do not list it.
- needs_external_services: true if running it needs a database, a network connection or files that do not exist yet."""

GEN_PROMPT = """Write the complete Java project for this lab question.

Question:
{question}

Requirements:
{requirements}"""

FIX_PROMPT = """This Java project failed a check.

Question:
{question}

Requirements:
{requirements}

Failed stage: {stage}
Problem:
{problem}

Current files:
{files}

Return ONLY the files you changed, with their FULL contents, in the same
===FILE:path=== ... ===END_FILE=== format. Fix the root cause and do not rewrite parts that already work."""

REVIEW_SYSTEM = "You are a strict Java code reviewer."
REVIEW_PROMPT = """Check this code against the lab question and requirements.

Question:
{question}

Requirements:
{requirements}

Files:
{files}

Sample runs (stdin -> stdout):
{transcript}

Reply with exactly OK if every requirement is met and the output is correct.
Otherwise list the concrete problems, one per line. Do not nitpick style."""

# ---------------------------------------------------------------- NetBeans (Ant) scaffold
PROJECT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://www.netbeans.org/ns/project/1">
    <type>org.netbeans.modules.java.j2seproject</type>
    <configuration>
        <data xmlns="http://www.netbeans.org/ns/j2se-project/3">
            <name>@NAME@</name>
            <source-roots>
                <root id="src.dir"/>
            </source-roots>
            <test-roots>
                <root id="test.src.dir"/>
            </test-roots>
        </data>
    </configuration>
</project>
"""

PROJECT_PROPERTIES = """annotation.processing.enabled=true
annotation.processing.enabled.in.editor=false
annotation.processing.processor.options=
annotation.processing.processors.list=
annotation.processing.run.all.processors=true
annotation.processing.source.output=${build.generated.sources.dir}/ap-source-output
application.title=@NAME@
application.vendor=student
build.classes.dir=${build.dir}/classes
build.classes.excludes=**/*.java,**/*.form
build.dir=build
build.generated.dir=${build.dir}/generated
build.generated.sources.dir=${build.dir}/generated-sources
build.sysclasspath=ignore
build.test.classes.dir=${build.dir}/test/classes
build.test.results.dir=${build.dir}/test/results
debug.classpath=${run.classpath}
debug.modulepath=${run.modulepath}
debug.test.classpath=${run.test.classpath}
debug.test.modulepath=${run.test.modulepath}
dist.archive.excludes=
dist.dir=dist
dist.jar=${dist.dir}/@NAME@.jar
dist.javadoc.dir=${dist.dir}/javadoc
dist.jlink.dir=${dist.dir}/jlink
dist.jlink.output=${dist.jlink.dir}/@NAME@
endorsed.classpath=
excludes=
includes=**
jar.compress=false
javac.classpath=
javac.compilerargs=
javac.deprecation=false
javac.external.vm=true
javac.modulepath=
javac.processormodulepath=
javac.processorpath=${javac.classpath}
javac.source=@RELEASE@
javac.target=@RELEASE@
javac.test.classpath=${javac.classpath}:${build.classes.dir}
javac.test.modulepath=${javac.modulepath}
javac.test.processorpath=${javac.test.classpath}
javadoc.additionalparam=
javadoc.author=false
javadoc.encoding=${source.encoding}
javadoc.html5=false
javadoc.noindex=false
javadoc.nonavbar=false
javadoc.notree=false
javadoc.private=false
javadoc.splitindex=true
javadoc.use=true
javadoc.version=false
javadoc.windowtitle=
jlink.additionalmodules=
jlink.additionalparam=
jlink.launcher=false
jlink.launcher.name=@NAME@
main.class=@MAIN@
manifest.file=manifest.mf
meta.inf.dir=${src.dir}/META-INF
mkdist.disabled=false
platform.active=default_platform
run.classpath=${javac.classpath}:${build.classes.dir}
run.jvmargs=
run.modulepath=${javac.modulepath}
run.test.classpath=${javac.test.classpath}:${build.test.classes.dir}
run.test.modulepath=${javac.test.modulepath}
source.encoding=UTF-8
src.dir=src
test.src.dir=test
"""

BUILD_XML = """<?xml version="1.0" encoding="UTF-8"?>
<project name="@NAME@" default="default" basedir=".">
    <description>Builds, tests, and runs the project @NAME@.</description>
    <import file="nbproject/build-impl.xml"/>
</project>
"""

MANIFEST_MF = "Manifest-Version: 1.0\nX-COMMENT: Main-Class will be added automatically by build\n\n"


def generated_scaffold(name, main):
    fill = lambda t: t.replace("@NAME@", name).replace("@MAIN@", main).replace("@RELEASE@", JAVA_RELEASE)
    return {
        "nbproject/project.xml": fill(PROJECT_XML),
        "nbproject/project.properties": fill(PROJECT_PROPERTIES),
        "build.xml": fill(BUILD_XML),
        "manifest.mf": MANIFEST_MF,
    }


def template_scaffold(tpl, name, main):
    """Use an nbproject/ folder exported from the user's own NetBeans. Only project.xml and
    project.properties are edited; build-impl.xml / genfiles.properties / build.xml are copied
    untouched because NetBeans checks them against stored checksums."""
    pxml = (tpl / "nbproject" / "project.xml").read_text(encoding="utf-8")
    old = re.search(r"<name>([^<]+)</name>", pxml).group(1)
    out = {}
    for f in (tpl / "nbproject").rglob("*"):
        rel = f.relative_to(tpl)
        if not f.is_file() or "private" in rel.parts:
            continue
        text = f.read_text(encoding="utf-8")
        key = rel.as_posix()
        if key == "nbproject/project.xml":
            text = text.replace(f"<name>{old}</name>", f"<name>{name}</name>")
        elif key == "nbproject/project.properties":
            text = text.replace(old, name)
            text = re.sub(r"(?m)^main\.class=.*$", lambda m: f"main.class={main}", text)
        out[key] = text
    for extra in ("build.xml", "manifest.mf"):
        if (tpl / extra).is_file():
            out[extra] = (tpl / extra).read_text(encoding="utf-8")
    return out


def scaffold(name, main):
    tpl = Path(NB_TEMPLATE_DIR)
    if (tpl / "nbproject" / "project.xml").is_file():
        return template_scaffold(tpl, name, main)
    return generated_scaffold(name, main)


# ---------------------------------------------------------------- parsing AI output
FILE_RE = re.compile(
    r"={3}\s*FILE:\s*(.+?)\s*={3}[ \t]*\r?\n(.*?)\r?\n?[ \t]*={3}\s*END_FILE\s*={3}", re.S)


def strip_fences(text):
    text = text.strip()
    text = re.sub(r"^```\w*[ \t]*\r?\n", "", text)
    text = re.sub(r"\r?\n```[ \t]*$", "", text)
    return text.strip()


def normalize_path(path, content):
    """Build the path from the code itself (package line + public class) instead of trusting the
    model: no 'src/src/...', no '..', and the folder always matches the package."""
    base = os.path.basename(path.strip().replace("\\", "/"))
    m = re.search(r"public\s+(?:(?:final|abstract|sealed)\s+)*(?:class|interface|enum|record)\s+(\w+)", content)
    if m:
        base = m.group(1) + ".java"
    pkg = re.search(r"^\s*package\s+([\w.]+)\s*;", content, re.M)
    return (pkg.group(1).replace(".", "/") + "/" if pkg else "") + base


def parse_project_files(ai_response, strict=False):
    """Parse model output into {relative_path: source}. strict=True only accepts ===FILE:...=== blocks
    (used for fix rounds, where a prose reply must not be mistaken for code)."""
    files = {}
    for path, content in FILE_RE.findall(ai_response):
        content = strip_fences(content)
        if path.strip().endswith(".java") and content:
            files[normalize_path(path, content)] = content + "\n"
    if files or strict:
        return files

    # fallbacks kept from the original script
    for path, content in re.findall(r"```(?:java)?\s*\n// (?:File: )?(.+?\.java)\s*\n(.*?)```", ai_response, re.S):
        content = content.strip()
        if content:
            files[normalize_path(path, content)] = content + "\n"
    if files:
        return files

    clean = strip_fences(ai_response)
    if not re.search(r"\b(class|interface|enum)\s+\w+", clean):
        return {}  # prose, not code
    cls = re.search(r"\bclass\s+(\w+)", clean)
    files[normalize_path(f"{cls.group(1) if cls else 'Main'}.java", clean)] = clean + "\n"
    return files


def find_main_class(files):
    """Fully qualified name of the class that contains main()."""
    for path, content in files.items():
        mm = re.search(r"static\s+void\s+main\s*\(", content)
        if not mm:
            continue
        classes = list(re.finditer(r"\bclass\s+(\w+)", content[:mm.start()]))
        cls = classes[-1].group(1) if classes else Path(path).stem
        pkg = re.search(r"^\s*package\s+([\w.]+)\s*;", content, re.M)
        return f"{pkg.group(1)}.{cls}" if pkg else cls
    return "Main"


def show(files):
    return "\n".join(f"===FILE:{p}===\n{c}===END_FILE===" for p, c in files.items())


def parse_json(text):
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


# ---------------------------------------------------------------- build + checks
def write_project(root, files, name, main):
    shutil.rmtree(root / "src", ignore_errors=True)
    for rel, text in scaffold(name, main).items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    for path, content in files.items():
        f = root / "src" / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
    (root / "test").mkdir(exist_ok=True)


def compile_project(root):
    out = root / "build" / "classes"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    sources = [str(p) for p in (root / "src").rglob("*.java")]
    if not sources:
        return False, "No .java files were produced."
    r = subprocess.run(
        ["javac", "--release", JAVA_RELEASE, "-encoding", "UTF-8", "-d", str(out), *sources],
        capture_output=True, text=True, timeout=90,
    )
    return r.returncode == 0, (r.stderr or r.stdout)


def run_checks(root, plan, main):
    """Run the compiled program. On success the returned text is a transcript for the reviewer."""
    out = root / "build" / "classes"
    if plan.get("needs_external_services"):
        return True, "(needs a database / network - compiled only, not run)"

    if plan.get("type") == "gui":
        xvfb = shutil.which("xvfb-run")
        if not xvfb:
            return True, "(GUI program - compiled only; xvfb-run not installed)"
        try:
            r = subprocess.run([xvfb, "-a", "timeout", "6", "java", "-cp", str(out), main],
                               capture_output=True, text=True, timeout=45)
        except subprocess.TimeoutExpired:
            return True, "(GUI program started and stayed open)"
        if r.returncode in (0, 124) and "Exception" not in r.stderr:  # 124 = still open when timeout hit
            return True, "(GUI program started and stayed open)" if r.returncode == 124 \
                else "(GUI program ran and exited cleanly)"
        return False, f"GUI program failed to start (exit {r.returncode}):\n{r.stderr[-1500:]}"

    code = "".join(p.read_text(encoding="utf-8") for p in (root / "src").rglob("*.java"))
    inputs = plan.get("test_inputs") or []
    if not inputs:
        if re.search(r"System\.in|Scanner|BufferedReader", code):
            return True, "(program reads input and no test input was planned - not run)"
        inputs = [""]
    transcript = []
    for text in inputs:
        try:
            r = subprocess.run(["java", "-cp", str(out), main], input=text,
                               capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            return False, f"Program hung or timed out with this stdin:\n{text}"
        if r.returncode != 0 or "Exception" in r.stderr:
            return False, f"Runtime failure with this stdin:\n{text}\nstderr:\n{r.stderr[-1500:]}"
        transcript.append(f"stdin: {text!r}\nstdout:\n{r.stdout[-1500:]}")
    return True, "\n---\n".join(transcript)


# ---------------------------------------------------------------- main entry
def build_project(project_name, question, call_ai, log=print, allow_partial=True):
    """Returns (zip_bytes, report); report = {"ok", "verified", "notes"}.
    If the loop cannot get a clean build and allow_partial is True, the best attempt is still zipped
    with a BUILD_NOTES.txt explaining what is left. Raises RuntimeError only if nothing usable
    could be produced at all (no plan / no files)."""
    if not (shutil.which("javac") and shutil.which("java")):
        raise RuntimeError("javac/java not found - install a JDK (not just a JRE) and put it on PATH")
    name = re.sub(r"[^A-Za-z0-9_]", "", project_name or "") or "JavaProject"

    # 1. plan (retry on bad JSON)
    plan = None
    for _ in range(3):
        try:
            plan = parse_json(call_ai(PLAN_PROMPT.format(question=question), PLAN_SYSTEM, "planner"))
            break
        except Exception as e:
            log(f"  plan retry: {e}")
    if plan is None:
        raise RuntimeError("planner never returned valid JSON")
    requirements = "\n".join(f"- {r}" for r in plan.get("requirements", []))

    libs = [str(x) for x in (plan.get("needs_libraries") or [])]
    verify = not libs  # cannot compile what needs jars we don't have
    notes = []
    if libs:
        notes.append("This project needs third-party libraries: " + ", ".join(libs) + ".\n"
                     "Add them in NetBeans (right-click Libraries > Add JAR/Folder, or use a Maven project). "
                     "It was NOT compiled automatically.")
    if plan.get("needs_external_services"):
        notes.append("Running it needs a database or network. Add the JDBC driver jar under Libraries "
                     "and edit the connection settings in the code. It was compiled but not run.")

    # 2. generate
    files = {}
    for _ in range(3):
        files = parse_project_files(call_ai(
            GEN_PROMPT.format(question=question, requirements=requirements), PROJECT_SYSTEM_PROMPT, "coder"))
        if files:
            break
    if not files:
        raise RuntimeError("coder never produced parseable files")

    # 3. compile -> run -> review -> fix loop
    tmp = tempfile.mkdtemp(prefix="moodai_")
    root = Path(tmp) / name
    root.mkdir()
    reviews_left = MAX_REVIEWS
    final_ok = True
    try:
        for attempt in range(MAX_FIX_ROUNDS + 1):
            main = find_main_class(files)
            write_project(root, files, name, main)
            if not verify:
                break
            stage = "compile"
            ok, problem = compile_project(root)
            if ok:
                stage = "run"
                ok, problem = run_checks(root, plan, main)
            if ok:
                transcript = problem
                if reviews_left == 0:
                    break
                reviews_left -= 1
                stage = "review"
                verdict = call_ai(REVIEW_PROMPT.format(
                    question=question, requirements=requirements,
                    files=show(files), transcript=transcript), REVIEW_SYSTEM, "reviewer").strip()
                if verdict.upper().startswith("OK"):
                    break
                problem = verdict
            if attempt == MAX_FIX_ROUNDS:
                if stage == "review":
                    break  # it compiles and runs; ship it rather than fail
                if not allow_partial:
                    raise RuntimeError(f"gave up after {attempt} fix rounds; last {stage} problem:\n{problem[:500]}")
                final_ok = False
                notes.insert(0, "The automatic build could not fully fix this project - open it in NetBeans "
                                f"and check. The {stage} step was still failing after {attempt} fix rounds:\n"
                                + problem[-2000:])
                break
            log(f"  fix round {attempt + 1} ({stage})")
            changed = parse_project_files(call_ai(FIX_PROMPT.format(
                question=question, requirements=requirements, stage=stage,
                problem=problem[-3000:], files=show(files)), PROJECT_SYSTEM_PROMPT, "fixer"), strict=True)
            for new_path in changed:  # a moved class must not leave its old copy behind
                for old_path in list(files):
                    if old_path != new_path and os.path.basename(old_path) == os.path.basename(new_path):
                        del files[old_path]
            files.update(changed)

        if notes:
            (root / "BUILD_NOTES.txt").write_text("\n\n".join(notes) + "\n", encoding="utf-8")

        # 4. zip (skip build output so NetBeans rebuilds it)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                rel = p.relative_to(root)
                if p.is_file() and rel.parts[0] not in ("build", "dist"):
                    z.write(p, str(Path(name) / rel))
            z.writestr(f"{name}/test/", "")
        return buf.getvalue(), {"ok": final_ok, "verified": verify, "notes": notes}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

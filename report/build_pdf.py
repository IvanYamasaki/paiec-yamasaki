"""draft.md -> build/report.tex -> report.pdf with the COLM 2026 style (preprint option, author shown).

No pandoc on this machine, so this converts only the Markdown that draft.md uses:
headings, paragraphs, lists, pipe tables, one code block, one image, inline code and URLs.

    python report/build_pdf.py            (then pdflatex runs twice inside report/build/)
    python report/build_pdf.py submission (anonymous version)
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
MODE = sys.argv[1] if len(sys.argv) > 1 else "preprint"
REPO_URL = "[TBD: public repository URL]"  # set when the GitHub repository is created

UNI = {
    "θ": r"$\theta$", "μ": r"$\mu$", "λ": r"$\lambda$", "≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\rightarrow$",
    "±": r"$\pm$", "√": r"$\surd$", "·": r"$\cdot$", "–": "--", "×": r"$\times$", " ": " ",
    " ": "~",
}
ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
       "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s):
    s = "".join(ESC.get(c, c) for c in s)
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s


def inline(s):
    out, pos = [], 0
    for m in re.finditer(r"`([^`]+)`|(https?://[^\s)]+[^\s).,;])|\*\*([^*]+)\*\*", s):
        out.append(esc(s[pos:m.start()]))
        if m.group(1):
            out.append(r"\texttt{" + esc(m.group(1)).replace("/", r"/\allowbreak{}").replace("-", "-{}") + "}")
        elif m.group(2):
            out.append(r"\url{" + m.group(2).replace("%", r"\%").replace("#", r"\#") + "}")
        else:
            out.append(r"\textbf{" + esc(m.group(3)) + "}")
        pos = m.end()
    out.append(esc(s[pos:]))
    return "".join(out).replace('"', "''")


def fix_quotes(s):
    # opening straight quotes -> ``
    return re.sub(r"(^|[\s(\[])''", r"\1``", s)


def table(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head, body = rows[0], rows[2:]
    n = len(head)
    spec = "l" + "r" * (n - 1) if all(re.match(r"^[-0-9.±\[\], ]*$", r[-1] or "0") for r in body) else "l" * n
    t = [r"\begin{center}\small", r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
         r"\begin{tabular}{" + spec + "}", r"\toprule",
         " & ".join(fix_quotes(inline(c)) for c in head) + r" \\", r"\midrule"]
    for r in body:
        r = (r + [""] * n)[:n]
        t.append(" & ".join(fix_quotes(inline(c)) for c in r) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}}", r"\end{center}"]
    return "\n".join(t)


def convert(md):
    L = md.split("\n")
    title = L[0].lstrip("# ").strip()
    author = L[2].strip()
    i, out, para, lst = 3, [], [], None

    def flush():
        nonlocal para, lst
        if para:
            out.append(fix_quotes(inline(" ".join(para))) + "\n")
            para = []
        if lst:
            env = "enumerate" if lst[0] == "ol" else "itemize"
            out.append(r"\begin{%s}" % env + "".join(r"\item " + fix_quotes(inline(x)) + "\n" for x in lst[1])
                       + r"\end{%s}" % env + "\n")
            lst = None

    in_abs = False
    while i < len(L):
        ln = L[i]
        s = ln.strip()
        if s.startswith("Technical report for"):
            i += 1
            continue
        if s.startswith("```"):
            flush()
            j = i + 1
            while not L[j].strip().startswith("```"):
                j += 1
            out.append(r"\begin{Verbatim}[fontsize=\footnotesize]" + "\n" + "\n".join(L[i + 1:j]) + "\n" + r"\end{Verbatim}")
            i = j + 1
            continue
        if s.startswith("|"):
            flush()
            j = i
            while j < len(L) and L[j].strip().startswith("|"):
                j += 1
            out.append(table(L[i:j]))
            i = j
            continue
        m = re.match(r"^(#{2,3}) (.*)$", s)
        if m:
            flush()
            if in_abs:
                out.append(r"\end{abstract}")
                in_abs = False
            name = re.sub(r"^\d+(\.\d+)?\.?\s+", "", m.group(2))
            if name == "Abstract":
                out.append(r"\begin{abstract}")
                in_abs = True
            elif name == "References":
                out.append(r"\section*{References}")
            else:
                out.append((r"\section{" if m.group(1) == "##" else r"\subsection{") + esc(name) + "}")
            i += 1
            continue
        m = re.match(r"^!\[[^]]*\]\(([^)]+)\)", s)
        if m:
            flush()
            out.append(r"\begin{center}\includegraphics[width=0.72\linewidth]{../" + m.group(1) + r"}\end{center}")
            i += 1
            continue
        m = re.match(r"^(-|\d+\.) (.*)$", s)
        if m:
            if para:
                flush()
            kind = "ol" if m.group(1)[0].isdigit() else "ul"
            if lst is None:
                lst = (kind, [])
            lst[1].append(m.group(2))
            i += 1
            continue
        if not s:
            flush()
            i += 1
            continue
        if re.match(r"^(Table|Figure) \d+\.", s):
            flush()
            k = s.index(".")
            out.append(r"\noindent\textbf{" + esc(s[:k + 1]) + "} " + fix_quotes(inline(s[k + 1:].strip())) + "\n")
            i += 1
            continue
        if re.match(r"^\[\d+\] ", s):
            flush()
            out.append(r"\noindent " + fix_quotes(inline(s)) + r"\par\smallskip" + "\n")
            i += 1
            continue
        para.append(s)
        i += 1
    flush()
    body = "\n".join(out)
    return title, author, body


PREAMBLE = r"""\documentclass{article}
\usepackage[%s]{colm2026_conference}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{microtype,url,booktabs,graphicx,fancyvrb,amssymb}
\usepackage{hyperref}
%s
\definecolor{darkblue}{rgb}{0, 0, 0.5}
\hypersetup{colorlinks=true, citecolor=darkblue, linkcolor=darkblue, urlcolor=darkblue}
\title{%s}
\author{%s}
\begin{document}
\maketitle
%s
%s
\end{document}
"""


def main():
    md = open(os.path.join(HERE, "draft.md"), encoding="utf8").read()
    md = md.replace("{REPO_URL}", "anonymized for review" if MODE == "submission" else REPO_URL)
    if MODE == "submission":
        # COLM's submission option already hides \author; drop the team name and e-mail from the text too
        md = md.replace("Team Yamasaki", "Anonymous team").replace("ivangyamasaki@gmail.com", "")
    title, author, body = convert(md)
    author = re.sub(r"^Team Yamasaki\.\s*", "", author)
    name, _, rest = author.rstrip(".").partition(", ")
    aff, _, mail = rest.rpartition(", ")
    author_tex = esc(name) + r" \\ " + esc(aff) + r" \\ \texttt{" + esc(mail) + r"} \\ Team Yamasaki"
    os.makedirs(BUILD, exist_ok=True)
    for f in ("colm2026_conference.sty", "fancyhdr.sty", "natbib.sty"):
        shutil.copy(os.path.join(HERE, "colm_dl", f), BUILD)
    head = r"\lhead{PAIEC 2026 technical report}" if MODE != "submission" else ""
    tex = PREAMBLE % (MODE, "", esc(title), author_tex, head, body)
    open(os.path.join(BUILD, "report.tex"), "w", encoding="utf8").write(tex)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd=BUILD,
                           capture_output=True, text=True, errors="replace", timeout=300)
    errs = [ln for ln in r.stdout.splitlines() if ln.startswith("!")]
    print("pdflatex rc", r.returncode, "errors:", errs[:10])
    pdf = os.path.join(BUILD, "report.pdf")
    if os.path.exists(pdf):
        dst = os.path.join(HERE, "report.pdf" if MODE != "submission" else "report_anon.pdf")
        shutil.copy(pdf, dst)
        print("wrote", dst)


if __name__ == "__main__":
    main()

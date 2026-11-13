"""Build the jury PowerPoint from the speaking guide and the measured study."""

from __future__ import annotations

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT = r"c:\Users\HP\Downloads\Jury Presentation - ML-Based Developer Behavior and Process Bottleneck Analysis.pptx"

BG = "0B1020"
CARD = "121A33"
LINE = "1F2A4D"
TEAL = "2DD4BF"
WHITE = "F8FAFC"
MUTED = "94A3B8"
BODY = "E5E7EB"
AMBER = "F59E0B"
RED = "EF4444"


def rgb(value: str) -> RGBColor:
    value = value.lstrip("#")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def set_run(paragraph, content, size, bold, color, align=None):
    paragraph.alignment = align or PP_ALIGN.LEFT
    run = paragraph.add_run()
    run.text = content
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    run.font.name = "Calibri"
    return run


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
prs.core_properties.title = "ML-Based Developer Behavior and Process Bottleneck Analysis in CI/CD Pipelines"
prs.core_properties.author = "Vanshika Nimwal, Avantika Sharma, Archita Shree"
BLANK = prs.slide_layouts[6]
SLIDES: list = []


def new_slide(notes: str = ""):
    slide = prs.slides.add_slide(BLANK)
    SLIDES.append(slide)
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(BG)
    shape.line.fill.background()
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = rgb(TEAL)
    bar.line.fill.background()
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def card(slide, left, top, width, height, fill=CARD):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(LINE)
    shape.adjustments[0] = 0.08
    return shape


def write(slide, left, top, width, height, content, size=18, bold=False, color=WHITE, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.auto_size = None
    set_run(frame.paragraphs[0], content, size, bold, color, align)
    return box


def bullets(slide, left, top, width, height, items, size=16, color=BODY, spacing=8):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(spacing)
        text, item_size, item_bold, item_color = item if isinstance(item, tuple) else (item, size, False, color)
        set_run(paragraph, text, item_size, item_bold, item_color)
    return box


def kicker(slide, left, top, content):
    write(slide, left, top, 8, 0.32, content, 12, True, TEAL)


def title(slide, content, subtitle=""):
    write(slide, 0.5, 0.28, 12.3, 0.62, content, 30, True, WHITE)
    if subtitle:
        write(slide, 0.5, 0.92, 12.3, 0.4, subtitle, 16, False, MUTED)


def paint_cell(cell, content, size, bold, color, fill):
    cell.text = ""
    cell.fill.solid()
    cell.fill.fore_color.rgb = rgb(fill)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = cell.text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    set_run(paragraph, content, size, bold, color, PP_ALIGN.CENTER)
    cell.margin_left = Emu(60000)
    cell.margin_right = Emu(60000)


def footer_all():
    total = len(SLIDES)
    for index, slide in enumerate(SLIDES, start=1):
        write(slide, 0.45, 7.12, 9.5, 0.28, "Simulated proof of concept  ·  Scores are measured, not copied from the paper", 12, False, MUTED)
        write(slide, 10.6, 7.12, 2.2, 0.28, f"{index}   /   {total}", 12, False, MUTED, PP_ALIGN.RIGHT)


# 1 Title
s = new_slide(
    "Open by separating three things: the literature review, the conceptual framework, and the simulated prototype. "
    "Never describe PipelineIQ as proven on a live Jenkins, GitHub Actions, or GitLab pipeline."
)
write(s, 0.7, 1.35, 11.5, 0.35, "SHARDA UNIVERSITY  ·  OCTOBER 2026", 14, True, TEAL)
write(s, 0.7, 1.85, 12, 1.6, "Machine learning for developer behavior and process bottlenecks in CI/CD", 36, True)
write(s, 0.7, 3.55, 11.5, 0.7, "A structured review of 20 studies, a behaviour-aware framework, and PipelineIQ — a simulated proof of concept.", 20, False, BODY)
card(s, 0.7, 4.7, 3.6, 1.45)
write(s, 0.9, 4.88, 3.2, 0.35, "TEAM", 12, True, TEAL)
write(s, 0.9, 5.25, 3.2, 0.7, "Vanshika Nimwal\nAvantika Sharma\nArchita Shree", 14, False, WHITE)
card(s, 4.5, 4.7, 3.8, 1.45)
write(s, 4.7, 4.88, 3.4, 0.35, "SUPERVISOR", 12, True, TEAL)
write(s, 4.7, 5.35, 3.4, 0.5, "Ms. Harminder Kaur", 18, True, WHITE)
card(s, 8.5, 4.7, 4.1, 1.45)
write(s, 8.7, 4.88, 3.7, 0.35, "WHAT THIS IS", 12, True, TEAL)
write(s, 8.7, 5.28, 3.7, 0.65, "Review + framework\n+ simulated prototype", 16, False, WHITE)

# 2 Agenda
s = new_slide("Eight minutes: one minute opening, one minute problem, one minute method, one minute findings, one minute framework, two minutes PipelineIQ, one minute honesty.")
title(s, "What we will cover")
items = [
    ("01", "The problem", "Failure prediction does not say where a pipeline loses time."),
    ("02", "The review", "20 studies, six groups, four research questions."),
    ("03", "The framework", "Behaviour-aware, time-safe, and not a tool for judging people."),
    ("04", "PipelineIQ", "4,800 simulated runs, five models, a seven-page dashboard."),
]
for index, (num, head, body) in enumerate(items):
    left = 0.5 + (index % 2) * 6.4
    top = 1.7 + (index // 2) * 2.4
    card(s, left, top, 6.1, 2.15)
    write(s, left + 0.3, top + 0.28, 1.2, 0.45, num, 22, True, TEAL)
    write(s, left + 1.5, top + 0.32, 4.2, 0.45, head, 22, True)
    write(s, left + 0.3, top + 1.05, 5.5, 0.75, body, 16, False, BODY)

# 3 Basics
s = new_slide("CI is automatic build and test on every change. CD is automatic preparation or release. A bottleneck is a slow stage, not the same thing as a failed build.")
title(s, "CI/CD, in one minute", "A pipeline is the conveyor from a code change to a release.")
basics = [
    ("Pipeline", "Commit, review, build, test, package, deploy, monitor."),
    ("Bottleneck", "A stage where work piles up: slow review, slow tests, or a long approval."),
    ("Failure", "A yes or no recorded by the system. A bottleneck is gradual and needs a definition."),
    ("This study", "Learn from past runs which new runs are likely to be slow, and say why."),
]
for index, (head, body) in enumerate(basics):
    top = 1.6 + index * 1.25
    card(s, 0.5, top, 12.3, 1.12)
    write(s, 0.75, top + 0.16, 3.2, 0.75, head, 20, True, TEAL)
    write(s, 4.1, top + 0.22, 8.3, 0.7, body, 18, False, BODY)

# 4 Problem
s = new_slide("Research is split. Predicting that a build will fail does not say whether the wait is in review, test, queue, or deployment approval.")
title(s, "The problem", "Five research streams are rarely joined, and they do not share a definition of success.")
streams = ["Build-failure prediction", "Test prioritization", "Process mining", "Delivery metrics", "Human factors"]
for index, name in enumerate(streams):
    left = 0.5 + index * 2.52
    card(s, left, 1.7, 2.38, 1.35)
    write(s, left + 0.12, 2.05, 2.14, 0.8, name, 15, True, WHITE, PP_ALIGN.CENTER)
card(s, 0.5, 3.3, 12.3, 1.55)
write(s, 0.75, 3.48, 11.8, 0.35, "CENTRAL PROBLEM", 12, True, TEAL)
write(s, 0.75, 3.9, 11.8, 0.7, "There is no integrated account of where and why a pipeline loses time. A failure alarm is an early warning, not a diagnosis.", 18, False, BODY)
write(s, 0.5, 5.15, 12.3, 1.5, "Developer behavior — commits, review wait, recent success — can add context. It does not prove that a person caused the delay. Slow review may be workload, time zone, or policy.", 18, False, BODY)

# 5 RQs
s = new_slide("State the four questions, then say the review answers them and the prototype only illustrates one feasible path.")
title(s, "Four research questions")
questions = [
    ("RQ1", "Which machine-learning and analytical techniques are used in CI/CD research?"),
    ("RQ2", "Which technical, process, and developer-related data are used?"),
    ("RQ3", "How do predictive and process-oriented methods treat failures, delays, and bottlenecks?"),
    ("RQ4", "What limits and research opportunities appear when these streams are read together?"),
]
for index, (code, question) in enumerate(questions):
    top = 1.6 + index * 1.25
    card(s, 0.5, top, 12.3, 1.1)
    write(s, 0.75, top + 0.28, 1.3, 0.5, code, 20, True, TEAL)
    write(s, 2.2, top + 0.28, 10.2, 0.6, question, 18, False, WHITE)

# 6 Review method
s = new_slide("Say this before the jury does: it is a structured review, not a formal systematic review. No PRISMA diagram, no meta-analysis, no claim of being exhaustive.")
title(s, "How the review was done", "A structured literature review of 20 studies, 2014 to 2026. Not a PRISMA systematic review.")
stats = [("20", "studies"), ("13", "journal articles"), ("5", "conference papers"), ("2", "book chapters")]
for index, (value, label) in enumerate(stats):
    left = 0.5 + index * 3.2
    card(s, left, 1.7, 3.0, 1.7)
    write(s, left + 0.15, 1.85, 2.7, 0.7, value, 32, True, TEAL, PP_ALIGN.CENTER)
    write(s, left + 0.15, 2.6, 2.7, 0.5, label, 15, False, BODY, PP_ALIGN.CENTER)
write(s, 0.5, 3.7, 12.3, 0.4, "SOURCES", 12, True, TEAL)
write(s, 0.5, 4.15, 12.3, 0.8, "IEEE Xplore, ScienceDirect, SpringerLink, ACM Digital Library, Wiley Online Library, and Google Scholar.", 18, False, BODY)
card(s, 0.5, 5.15, 12.3, 1.55)
write(s, 0.75, 5.35, 11.8, 1.1, "More of the 20 studies appear from 2024 onward (4 in 2024, 4 in 2025, 3 in 2026). That is not a publication trend. The search was not exhaustive, and the search log was not preserved.", 18, False, BODY)

# 7 Six groups
s = new_slide("Accuracy cannot be ranked across these groups. They answer different questions.")
title(s, "Six groups of studies")
groups = [
    ("A", "7", "Build-failure prediction", "Will this build fail, and why?"),
    ("B", "2", "Test prioritization", "Which tests should run first?"),
    ("C", "3", "Process mining and monitoring", "Where does the real workflow wait?"),
    ("D", "3", "Delivery metrics", "Lead time, deploy frequency, restore time."),
    ("E", "3", "Human and organizational factors", "Collaboration and influence."),
    ("F", "2", "CI/CD adoption reviews", "What blocks teams from adopting CI/CD?"),
]
for index, (code, count, head, body) in enumerate(groups):
    left = 0.4 + (index % 3) * 4.25
    top = 1.65 + (index // 3) * 2.5
    card(s, left, top, 4.05, 2.25)
    write(s, left + 0.25, top + 0.2, 1.2, 0.4, code, 18, True, TEAL)
    write(s, left + 2.3, top + 0.2, 1.5, 0.4, count + " studies", 14, True, AMBER, PP_ALIGN.RIGHT)
    write(s, left + 0.25, top + 0.75, 3.55, 0.7, head, 18, True)
    write(s, left + 0.25, top + 1.5, 3.55, 0.5, body, 14, False, MUTED)

# 8 Findings
s = new_slide("Prediction is an early warning. A bottleneck is a hypothesis to check, not proof of cause.")
title(s, "What the review found")
findings = [
    ("RQ1  ·  Techniques", "Five families: failure prediction, test prioritization, process mining, monitoring and metrics, and human analysis. They play different roles."),
    ("RQ2  ·  Data", "Repository, CI, process, delivery, and human data all exist. The hard part is linking them in time. The person who commits is not always the person who fixes the build."),
    ("RQ3  ·  Two questions", "Failure prediction asks “will this run fail?” Bottleneck analysis asks “where does delay or rework build up?”"),
]
for index, (head, body) in enumerate(findings):
    top = 1.6 + index * 1.7
    card(s, 0.5, top, 12.3, 1.55)
    write(s, 0.75, top + 0.18, 11.8, 0.4, head, 16, True, TEAL)
    write(s, 0.75, top + 0.65, 11.8, 0.7, body, 16, False, BODY)

# 9 Gaps
s = new_slide("These four gaps are why the prototype is simulated and why we refuse to copy the paper's F1 scores.")
title(s, "Four gaps")
gaps = [
    ("01", "Results may not transfer", "Public GitHub projects and industry pipelines do not share datasets or prediction targets."),
    ("02", "Bottleneck labels are weak", "A failure is obvious. A bottleneck is gradual and depends on the definition you choose."),
    ("03", "Developer data is easy to misuse", "A slow response can be workload, time zone, or policy, not poor performance."),
    ("04", "Offline papers hide operations", "Explainability, leakage, drift, cost, and governance are rarely tested."),
]
for index, (num, head, body) in enumerate(gaps):
    left = 0.45 + (index % 2) * 6.4
    top = 1.65 + (index // 2) * 2.5
    card(s, left, top, 6.15, 2.25)
    write(s, left + 0.25, top + 0.22, 1.2, 0.4, num, 18, True, TEAL)
    write(s, left + 1.3, top + 0.25, 4.5, 0.4, head, 18, True)
    write(s, left + 0.25, top + 0.95, 5.65, 0.95, body, 16, False, BODY)

# 10 Framework
s = new_slide("The report framework is conceptual. PipelineIQ is one simulated illustration of the predictive half, not a full validation of every layer.")
title(s, "Behaviour-aware framework", "Decision support. Outputs are hypotheses, not verdicts, and never an automatic judgement of a person.")
layers = [
    ("1", "Sources", "Repository events and pipeline telemetry."),
    ("2", "Features", "Linked in time, and computable before the prediction point."),
    ("3", "Methods", "Prediction and process views, checked against a simple baseline."),
    ("4", "Meaning", "Explainable bottleneck hypotheses, not blame."),
    ("5", "Decision", "Delivery insight a team can act on."),
]
for index, (num, head, body) in enumerate(layers):
    left = 0.4 + index * 2.55
    card(s, left, 1.7, 2.42, 3.15)
    write(s, left + 0.15, 1.9, 2.1, 0.45, num, 20, True, TEAL, PP_ALIGN.CENTER)
    write(s, left + 0.15, 2.5, 2.1, 0.7, head, 18, True, WHITE, PP_ALIGN.CENTER)
    write(s, left + 0.15, 3.3, 2.1, 1.3, body, 13, False, MUTED, PP_ALIGN.CENTER)
write(s, 0.5, 5.15, 12.3, 1.5, "Leakage means using a fact that would not yet be known, such as the final duration of the run you are trying to predict. The framework cuts features off at a prediction point. The prototype enforces the same idea in code.", 16, False, BODY)

# 11 Data
s = new_slide("Say simulated every time. 4,800 unique runs. The raw file has 4,896 rows because about 2 percent are webhook duplicates, and those are removed before training.")
title(s, "PipelineIQ data", "Proof of concept. No public dataset links developer behavior to a run-level bottleneck label.")
stats = [("4,800", "unique runs"), ("12", "projects"), ("35%", "bottleneck rate"), ("960", "held-out test runs")]
for index, (value, label) in enumerate(stats):
    left = 0.45 + index * 3.2
    card(s, left, 1.65, 3.05, 1.55)
    write(s, left + 0.1, 1.78, 2.85, 0.65, value, 28, True, TEAL, PP_ALIGN.CENTER)
    write(s, left + 0.1, 2.5, 2.85, 0.45, label, 14, False, BODY, PP_ALIGN.CENTER)
card(s, 0.45, 3.45, 6.2, 3.2)
write(s, 0.7, 3.65, 5.7, 0.35, "THE LABEL", 12, True, TEAL)
write(s, 0.7, 4.1, 5.7, 2.2, "A run is a bottleneck when its total time exceeds 1.5 times that project’s own median. A naturally slow project is not punished. The median uses the whole project, so the label is retrospective.", 16, False, BODY)
card(s, 6.85, 3.45, 5.95, 3.2)
write(s, 7.1, 3.65, 5.5, 0.35, "EIGHT FEATURES", 12, True, TEAL)
write(s, 7.1, 4.1, 5.5, 2.2, "Commit frequency, code churn, review latency, recent build success, historical success, build duration, test duration, deployment delay.\n\nPlatforms: Jenkins, GitHub Actions, GitLab CI.", 16, False, BODY)

# 12 Leakage
s = new_slide("The test set is the exam. Imputation medians and the min-max scaler are fit on the 3,840 training rows only. The hybrid blender never sees test scores.")
title(s, "The test set never leaks into training")
steps = [
    ("Split first", "3,840 train and 960 test, stratified, seed 42, before impute or scale."),
    ("Duplicates", "Webhook retries removed. 4,896 raw rows become 4,800 unique runs."),
    ("Impute", "Missing cells filled with medians from the training rows only."),
    ("Scale", "Min-max scaler fit on the training split and saved. Clean data stays in original units."),
    ("Hybrid", "The blender is fit on out-of-fold training probabilities only."),
    ("Honest caveat", "The saved model is chosen on the seed-42 test F1, so that one number is mildly optimistic. Five seeds are the uncertainty check."),
]
for index, (head, body) in enumerate(steps):
    left = 0.4 + (index % 3) * 4.25
    top = 1.65 + (index // 3) * 2.5
    card(s, left, top, 4.05, 2.25)
    write(s, left + 0.22, top + 0.25, 3.6, 0.45, head, 16, True, TEAL)
    write(s, left + 0.22, top + 0.85, 3.6, 1.15, body, 14, False, BODY)

# 13 Models
s = new_slide("Five models. The live scorer is the SVM because it won seed-42 test F1. The hybrid is reported beside it, not hidden.")
title(s, "Five models", "Each base model is tuned by 5-fold stratified F1 search on the training split only.")
models = [
    ("Logistic regression", "Transparent linear baseline. Class weights are balanced."),
    ("SVM, RBF kernel", "A flexible boundary. This is the saved predictor."),
    ("Gradient boosting", "Trees in sequence, each correcting the last."),
    ("Random forest", "An average of trees. Also used for feature ranking."),
    ("Hybrid stack", "A logistic regression blends the four probabilities. Added after the first four-model draft."),
]
for index, (head, body) in enumerate(models):
    top = 1.55 + index * 1.02
    card(s, 0.5, top, 12.3, 0.92)
    write(s, 0.75, top + 0.14, 3.6, 0.6, head, 16, True, TEAL)
    write(s, 4.5, top + 0.18, 8.0, 0.55, body, 16, False, BODY)

# 14 Results
s = new_slide(
    "SVM 74.1 percent F1 is the live model. Hybrid is 73.1 on this split, 72.3 plus or minus 1.3 across five seeds, and has the highest ROC-AUC at 0.887. "
    "Paper random-forest F1 is 93.2 percent. We did not tune the simulator to chase it."
)
title(s, "Held-out results", "960 test runs. Accuracy is about 80 percent for every model. F1 is the ranking metric because the classes are uneven.")
rows = [
    ["Model", "F1", "ROC-AUC", "F1 across 5 seeds", "Paper F1"],
    ["Logistic regression", "73.6%", "0.879", "71.5% ± 2.1", "78.4%"],
    ["SVM (RBF)  ·  live", "74.1%", "0.880", "71.2% ± 2.3", "83.1%"],
    ["Gradient boosting", "70.5%", "0.883", "69.2% ± 2.5", "91.7%"],
    ["Random forest", "73.7%", "0.881", "71.4% ± 1.9", "93.2%"],
    ["Hybrid stack", "73.1%", "0.887", "72.3% ± 1.3", "—"],
]
table_shape = s.shapes.add_table(len(rows), 5, Inches(0.45), Inches(1.7), Inches(12.4), Inches(4.15))
table = table_shape.table
table.columns[0].width = Inches(3.5)
for col in range(1, 5):
    table.columns[col].width = Inches(2.225)
for r, row in enumerate(rows):
    for c, value in enumerate(row):
        fill = "12362E" if r == 2 else ("0E152B" if r == 0 else CARD)
        color = TEAL if r == 0 or (r == 2 and c == 0) else WHITE
        paint_cell(table.cell(r, c), value, 14 if r else 13, r == 0 or c == 0, color, fill)
write(s, 0.5, 6.05, 12.3, 0.85, "The hybrid is the most stable and the best ranker. The SVM wins the official split by one F1 point, so it stays live. That is a stated trade-off.", 16, False, BODY)

# 15 Importance
s = new_slide("Behavior carries the signal. Dropping build and test duration costs 0.9 F1 points. Behavior-only is 3.6 points behind the full SVM.")
title(s, "What the model uses", "Random-forest share of importance. Permutation on the SVM agrees that churn and review latency dominate.")
shares = [("74.6%", "Developer behavior", "Churn, review latency, historical success, recent success, commits."), ("18.4%", "Build and test time", "Useful, but not enough on their own."), ("7.0%", "Deployment delay", "Reported on its own, not folded into timing.")]
for index, (value, head, body) in enumerate(shares):
    left = 0.45 + index * 4.25
    card(s, left, 1.65, 4.05, 2.35)
    write(s, left + 0.2, 1.8, 3.65, 0.6, value, 28, True, TEAL)
    write(s, left + 0.2, 2.5, 3.65, 0.4, head, 16, True)
    write(s, left + 0.2, 3.0, 3.65, 0.75, body, 14, False, MUTED)
card(s, 0.45, 4.2, 12.4, 2.5)
write(s, 0.7, 4.4, 11.9, 0.35, "ABLATION  ·  SAME SVM  ·  TEST F1", 12, True, TEAL)
write(s, 0.7, 4.9, 11.9, 1.45, "All eight features  74.1%.\nWithout build and test duration  73.3%  (−0.9 points).\nBehavior only  70.5%  (−3.6 points).\nBuild duration looks important in tree importance and almost disappears under permutation (0.1%). That is a known bias, and we report both.", 16, False, BODY)

# 16 Detection
s = new_slide("Do not hide this. At tau 0.65 there are 10 alerts on the full timeline and zero on the held-out runs. The 8 percent false-alarm target is not met. That is a finding.")
title(s, "Alerts need two rules", "One hot run is not an alert. Both conditions must hold.")
rules = [
    ("Probability", "Model probability above 0.65."),
    ("Persistence", "A stage stays more than 2 standard deviations above its previous 20-run mean."),
    ("Streak", "That spike lasts at least 3 consecutive runs."),
]
for index, (head, body) in enumerate(rules):
    left = 0.45 + index * 4.25
    card(s, left, 1.65, 4.05, 1.7)
    write(s, left + 0.22, 1.82, 3.6, 0.4, head, 16, True, TEAL)
    write(s, left + 0.22, 2.3, 3.6, 0.8, body, 15, False, BODY)
card(s, 0.45, 3.6, 12.4, 3.05)
write(s, 0.7, 3.8, 11.9, 0.4, "WHAT HAPPENED", 12, True, AMBER)
write(s, 0.7, 4.3, 11.9, 2.0, "10 confirmed alerts on the full timeline. 0 on the 960 held-out runs, so the false-alarm rate is undefined and the target of under 8 percent is not met.\n\nAt a looser threshold of 0.50 there is 1 test alert and still almost no detection. The rule is strict because a stage rarely stays that hot. Most labelled bottlenecks never become confirmed stage alerts.", 16, False, BODY)

# 17 Dashboard
s = new_slide("Show Executive Overview, then Developer Behavior or the what-if, then Model Lab. Say the numbers are computed from the runs.")
title(s, "PipelineIQ dashboard", "Seven pages. Filters start blank, which means include everything.")
pages = [
    "Executive Overview — rate, F1, hours at risk, project league.",
    "Developer Behavior — heatmap, coaching note, interaction scatter.",
    "Stage Performance — build, unit test, integration, deploy.",
    "Alert Center — only dual-rule alerts.",
    "What-if — sliders on the saved SVM, not a lookup table.",
    "Model Lab — five models against the paper, importance, ablation.",
    "Architecture — five layers and the limits, said out loud.",
]
bullets(s, 0.55, 1.55, 7.3, 5.2, [(item, 15, False, BODY) for item in pages], spacing=6)
card(s, 8.05, 1.6, 4.8, 5.05)
write(s, 8.25, 1.8, 4.4, 0.4, "AFTER THE FIRST DRAFT", 12, True, TEAL)
write(s, 8.25, 2.35, 4.4, 4.0, "The first sidebar pre-selected every developer and project, so it was a wall of tags.\n\nCharts fought the dark theme.\n\nCards broke out of their grids.\n\nThe menu was an embedded panel with a white box.\n\nFilters now have their own section, and the what-if presets sit on one row.", 14, False, BODY)

# 18 Real data
s = new_slide("We checked a real GitHub account and rejected it as a replacement. Large counts were Lock Threads and No Response bots. PayFlow had 14 real CI runs.")
title(s, "We checked real GitHub data", "It cannot replace the 4,800-run study. The dashboard still shows the simulation.")
checks = [
    ("13,101", "Ferrero runs", "Scheduled bots. They finish in a few seconds. Zero push builds."),
    ("14", "PayFlow CI runs", "The only real build-and-test workflow. Far too small."),
    ("164", "Security checks", "One workflow, mostly failures, no review time or stage split."),
]
for index, (value, head, body) in enumerate(checks):
    left = 0.45 + index * 4.25
    card(s, left, 1.7, 4.05, 2.45)
    write(s, left + 0.2, 1.88, 3.65, 0.6, value, 28, True, TEAL)
    write(s, left + 0.2, 2.6, 3.65, 0.4, head, 16, True)
    write(s, left + 0.2, 3.1, 3.65, 0.8, body, 14, False, MUTED)
write(s, 0.5, 4.45, 12.3, 2.2, "The code is on GitHub: vanshikanimwal-dev/FINAL-YEAR-PROJECT-. A root streamlit_app.py was added so the dashboard can be hosted. Hosting still needs a Streamlit Cloud login. Locally it runs with streamlit run streamlit_app.py.", 18, False, BODY)

# 19 Contribution
s = new_slide("Contribution is the comparison, the framework, the evaluation agenda, and a working simulated prototype. Next step is real timestamped pipeline data with the label fixed first.")
title(s, "Contribution and what comes next")
card(s, 0.45, 1.65, 6.2, 5.0)
write(s, 0.7, 1.85, 5.7, 0.4, "CONTRIBUTION", 12, True, TEAL)
bullets(
    s,
    0.7,
    2.4,
    5.7,
    3.9,
    [
        "A structured comparison of five research streams.",
        "A clear split between build outcomes and end-to-end bottlenecks.",
        "A behaviour-aware framework and an evaluation agenda.",
        "A working simulated prototype with an honest model comparison.",
    ],
    16,
)
card(s, 6.85, 1.65, 6.0, 5.0)
write(s, 7.1, 1.85, 5.5, 0.4, "NEXT", 12, True, TEAL)
bullets(
    s,
    7.1,
    2.4,
    5.5,
    3.9,
    [
        "Collect real timestamped pipeline logs.",
        "Fix the bottleneck definition before tuning.",
        "Split by time and test across projects.",
        "Check explanations with practitioners.",
        "Keep model output out of personnel decisions.",
    ],
    16,
)

# 20 Close
s = new_slide("Golden rule: literature review, conceptual framework, and simulated proof of concept stay separate. Do not say the results are proven on real pipelines.")
write(s, 0.7, 1.7, 12, 0.4, "THE LINE TO CLOSE ON", 14, True, TEAL)
write(s, 0.7, 2.3, 12, 2.2, "We can rank a risky run from developer behavior before the log turns red. On this simulation the SVM does that at 74.1 percent F1. We will not claim more than the evidence allows.", 28, True)
card(s, 0.7, 4.9, 11.9, 1.7)
write(s, 0.95, 5.15, 11.4, 1.2, "Review of 20 studies. A framework that treats predictions as hypotheses. A simulated prototype whose scores were measured, including the result that the strict alert rule did not fire on the test set.", 18, False, BODY)

# 21 Backup questions
s = new_slide("Use this only if asked. Short answers are in the speaker notes of this slide as well as on the slide.")
title(s, "If the jury asks", "Short answers. The full script is in the speaker notes.")
qa = [
    ("Why simulated?", "No public labelled set. A real GitHub export was bots plus 14 builds."),
    ("Is the model cheating?", "Split first. Impute and scale on train only. Hybrid uses out-of-fold scores."),
    ("Why below the paper?", "Own simulation, strict leakage control. Figures are not comparable."),
    ("Why not the hybrid?", "SVM wins the official F1, 74.1 against 73.1. Hybrid is more stable."),
    ("Is developer data ethical?", "Context, not blame. Never an automatic personnel judgement."),
    ("What is a bottleneck here?", "Total time over 1.5× that project’s own median. One definition among several."),
]
for index, (q, a) in enumerate(qa):
    left = 0.4 + (index % 3) * 4.25
    top = 1.6 + (index // 3) * 2.55
    card(s, left, top, 4.05, 2.35)
    write(s, left + 0.2, top + 0.2, 3.65, 0.7, q, 15, True, TEAL)
    write(s, left + 0.2, top + 0.95, 3.65, 1.15, a, 14, False, BODY)

footer_all()
prs.save(OUT)
print(OUT)
print(len(SLIDES))

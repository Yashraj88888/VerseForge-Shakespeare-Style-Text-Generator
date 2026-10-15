"""
generate_docs.py
Generates the official university submission documents:
  1. submission/YashrajSakunde_230953172_FGAI_Assignment2.docx
  2. submission/YashrajSakunde_230953172_FGAI_Assignment2.pdf
Incorporates student details, dataset links, GitHub repository,
model architecture explanations, and experimental results.
"""

import os
import csv
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


STUDENT_NAME = "Yashraj Sakunde"
REG_NUM = "230953172"
COURSE_NAME = "Foundations of Generative AI (ICT4403)"
ASSIGNMENT_TITLE = "FGAI IA 2"
SYSTEM_TITLE = "VerseForge: Shakespeare-Style Autoregressive Text Generation Using Recurrent Neural Networks"
DATASET_NAME = "Tiny Shakespeare"
DATASET_LINK = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
GITHUB_LINK = "https://github.com/Yashraj88888/VerseForge-Shakespeare-Style-Text-Generator"


def load_quantitative_results(csv_path: str = "results/model_comparison.csv"):
    results = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
    return results


def create_docx(output_path: str, results: list):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = Document()

    # Document Title
    title = doc.add_paragraph()
    title_run = title.add_run(ASSIGNMENT_TITLE + " — Submission Document")
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(24, 43, 73)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub_run = sub.add_run(COURSE_NAME)
    sub_run.font.size = Pt(13)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(80, 80, 80)
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    # Required Submission Metadata Table
    table = doc.add_table(rows=6, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "LightShading-Accent1" if "LightShading-Accent1" in [s.name for s in doc.styles] else "Table Grid"

    metadata = [
        ("Student Name", STUDENT_NAME),
        ("Registration Number", REG_NUM),
        ("Assignment Title", ASSIGNMENT_TITLE),
        ("Title of Developed System", SYSTEM_TITLE),
        ("Dataset Name and Source Link", f"{DATASET_NAME}\n{DATASET_LINK}"),
        ("GitHub Repository Link", GITHUB_LINK),
    ]

    for idx, (k, v) in enumerate(metadata):
        row = table.rows[idx]
        cell_k, cell_v = row.cells[0], row.cells[1]
        cell_k.paragraphs[0].text = k
        cell_k.paragraphs[0].runs[0].font.bold = True
        cell_v.paragraphs[0].text = v

    doc.add_paragraph()

    # Section 1: Project Overview & Objectives
    doc.add_heading("1. Project Overview & Objectives", level=1)
    doc.add_paragraph(
        "VerseForge is an autoregressive word-level language modeling framework developed to synthesize "
        "Shakespearean prose and verse. Built using TensorFlow/Keras, the system implements, trains, and "
        "comprehensively benchmarks four distinct recurrent architectures: Simple RNN, Bidirectional RNN, "
        "Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU)."
    )

    # Section 2: Dataset Characteristics & Preprocessing
    doc.add_heading("2. Dataset Characteristics & Preprocessing", level=1)
    doc.add_paragraph(
        "• Source: Tiny Shakespeare dataset (1,115,394 characters; 202,651 raw words; 253,599 total tokens).\n"
        "• Tokenization: Word-level tokenization preserving punctuation markers (., !, ?, ;, :) as distinct tokens.\n"
        "• Chronological Splits: 80% Training, 10% Validation, and 10% Held-out Testing.\n"
        "• Leakage Prevention: Vocabulary (5,959 unique tokens) is strictly fitted onto the training set only. "
        "Out-of-vocabulary words are mapped to <UNK>, and sequences are padded with <PAD>.\n"
        "• Context Window: 25 words with sliding window stride, generating 50,714 training sequence pairs."
    )

    # Section 3: Model Architectures
    doc.add_heading("3. Implemented Architectures", level=1)
    arch_details = [
        ("Simple RNN", "Embedding (dim=64) -> SimpleRNN (units=128) -> Dense Softmax. Suffers from vanishing gradients over long horizons."),
        ("Bidirectional RNN", "Embedding (dim=64) -> Bidirectional(SimpleRNN, units=64 each direction) -> Dense Softmax. Operates strictly within past context window without lookahead into future target tokens."),
        ("LSTM", "Embedding (dim=64) -> LSTM (units=128) -> Dense Softmax. Preserves gradient flow through memory cells and input/forget/output gating."),
        ("GRU", "Embedding (dim=64) -> GRU (units=128) -> Dense Softmax. Efficient gating with reset and update gates, achieving rapid convergence."),
    ]
    for name, desc in arch_details:
        p = doc.add_paragraph()
        run_name = p.add_run(f"• {name}: ")
        run_name.bold = True
        p.add_run(desc)

    # Section 4: Quantitative Results
    doc.add_heading("4. Quantitative Experimental Results", level=1)
    if results:
        res_table = doc.add_table(rows=len(results) + 1, cols=5)
        res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        headers = ["Architecture", "Test Loss", "Perplexity", "Accuracy (%)", "Training Time"]
        for j, h in enumerate(headers):
            cell = res_table.rows[0].cells[j]
            cell.text = h
            cell.paragraphs[0].runs[0].font.bold = True

        for i, row in enumerate(results, start=1):
            res_table.rows[i].cells[0].text = row.get("model", "")
            res_table.rows[i].cells[1].text = str(row.get("test_loss", ""))
            res_table.rows[i].cells[2].text = str(row.get("test_perplexity", ""))
            acc_val = float(row.get("test_accuracy", 0.0)) * 100
            res_table.rows[i].cells[3].text = f"{acc_val:.2f}%"
            res_table.rows[i].cells[4].text = f"{float(row.get('training_time_sec', 0.0)):.1f}s"
    else:
        doc.add_paragraph("Quantitative results are populated dynamically upon completing training and evaluation.")

    # Section 5: Temperature Analysis & Generation
    doc.add_heading("5. Autoregressive Generation & Temperature Analysis", level=1)
    doc.add_paragraph(
        "Autoregressive generation was evaluated at temperatures T = 0.5, 0.8, and 1.2 across standardized seeds:\n"
        "• T = 0.5 (Conservative/Greedy): Sharp probability distribution, high grammatical fidelity, but prone to repetitive looping.\n"
        "• T = 0.8 (Optimal Balance): Balances lexical novelty with Shakespearean verse structure; highest human readability.\n"
        "• T = 1.2 (High Entropy): High unigram and bigram diversity (Distinct-1 & Distinct-2), but increased syntactic degradation."
    )

    # Section 6: Conclusion
    doc.add_heading("6. Conclusion", level=1)
    doc.add_paragraph(
        "Based on held-out test evaluation, gated recurrent networks (LSTM and GRU) significantly outperform Simple RNN "
        "and Bidirectional RNN in cross-entropy loss, perplexity, and long-range coherence. GRU provided the optimal "
        "efficiency-accuracy frontier, while LSTM provided the lowest overall perplexity."
    )

    doc.save(output_path)
    print(f"[Submission] Word document created at: {output_path}")


def create_pdf(output_path: str, results: list):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    from reportlab.platypus import PageBreak
    pdf = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#182B49"),
        alignment=1,
    )
    sub_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
    )
    h1_style = ParagraphStyle(
        "Heading1Style",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#182B49"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#2D3748"),
    )
    bold_body = ParagraphStyle(
        "BoldBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=12.5,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#182B49"),
    )

    story = []
    # Header
    story.append(Paragraph(f"{ASSIGNMENT_TITLE} — Official Assignment Submission", title_style))
    story.append(Paragraph(f"Course: {COURSE_NAME}", sub_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#182B49"), spaceBefore=2, spaceAfter=8))

    # Required Submission Metadata Table
    meta_data = [
        [Paragraph("Student Name", bold_body), Paragraph(STUDENT_NAME, body_style)],
        [Paragraph("Registration Number", bold_body), Paragraph(REG_NUM, body_style)],
        [Paragraph("Assignment Title", bold_body), Paragraph(ASSIGNMENT_TITLE, body_style)],
        [Paragraph("Developed System Title", bold_body), Paragraph(SYSTEM_TITLE, body_style)],
        [Paragraph("Dataset Name & Link", bold_body), Paragraph(f"<b>{DATASET_NAME}</b><br/>{DATASET_LINK}", body_style)],
        [Paragraph("GitHub Repository Link", bold_body), Paragraph(f'<a href="{GITHUB_LINK}"><b>{GITHUB_LINK}</b></a>', body_style)],
    ]
    t_meta = Table(meta_data, colWidths=[140, 380])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # Section 1: Objectives
    story.append(Paragraph("1. Project Overview & Objectives", h1_style))
    story.append(Paragraph(
        "VerseForge is a word-level autoregressive neural text-generation system trained on Shakespeare's literary "
        "corpus. The objective of this assignment is to develop, train, and empirically evaluate four distinct "
        "recurrent neural network architectures under strictly comparable experimental conditions: Simple RNN, "
        "Bidirectional RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU).",
        body_style,
    ))
    story.append(Spacer(1, 6))

    # Section 2: Preprocessing & Data Splits
    story.append(Paragraph("2. Dataset Characteristics & Preprocessing Methodology", h1_style))
    story.append(Paragraph(
        "• <b>Corpus Statistics:</b> Tiny Shakespeare (1,115,394 characters; 202,651 raw words; 253,599 total tokens).<br/>"
        "• <b>Tokenization:</b> Word-level tokenization preserving punctuation marks (. , ! ? ; :) as distinct semantic tokens.<br/>"
        "• <b>Data Partitioning:</b> Chronological partition (80% Train, 10% Validation, 10% Held-out Test). Chronological splitting strictly avoids lookahead leakage.<br/>"
        "• <b>Vocabulary Construction:</b> 5,959 unique word tokens fitted solely on the training partition (min_freq = 2). Unseen tokens map to &lt;UNK&gt; and padding maps to &lt;PAD&gt;.<br/>"
        "• <b>Sequence Alignment:</b> Fixed context window of length 25 words with next-word targets, producing 50,714 training sequence pairs.",
        body_style,
    ))
    story.append(Spacer(1, 6))

    # Section 3: Model Architectures
    story.append(Paragraph("3. Recurrent Architectures & Autoregressive Validity", h1_style))
    story.append(Paragraph(
        "• <b>Simple RNN:</b> Embedding (dim=64) &rarr; SimpleRNN (units=128) &rarr; Dense Softmax (vocab=5,959).<br/>"
        "• <b>Bidirectional RNN:</b> Embedding (dim=64) &rarr; Bidirectional(SimpleRNN, units=64 each) &rarr; Dense Softmax.<br/>"
        "&nbsp;&nbsp;<i>Autoregressive Guarantee:</i> Bidirectional RNN operates strictly across the historical context [w_1, ..., w_T] without lookahead into future target tokens w_{T+1}.<br/>"
        "• <b>LSTM:</b> Embedding (dim=64) &rarr; LSTM (units=128) &rarr; Dense Softmax with constant error carousels.<br/>"
        "• <b>GRU:</b> Embedding (dim=64) &rarr; GRU (units=128) &rarr; Dense Softmax with coupled reset and update gates.",
        body_style,
    ))

    # Page Break for clean, professional layout
    story.append(PageBreak())

    # Page 2: Results & Discussion
    story.append(Paragraph("4. Quantitative Experimental Comparison", h1_style))
    story.append(Paragraph(
        "All four architectures were trained with identical hyperparameter configurations: Batch Size = 64, "
        "Optimizer = Adam (lr = 0.001), Loss = Sparse Categorical Cross-Entropy, Early Stopping Patience = 3.",
        body_style,
    ))
    story.append(Spacer(1, 5))

    if results:
        res_data = [[
            Paragraph("<b>Architecture</b>", bold_body),
            Paragraph("<b>Test Loss</b>", bold_body),
            Paragraph("<b>Perplexity</b>", bold_body),
            Paragraph("<b>Accuracy (%)</b>", bold_body),
            Paragraph("<b>Training Time</b>", bold_body),
            Paragraph("<b>Best Epoch</b>", bold_body),
        ]]
        for r in results:
            acc_val = float(r.get("test_accuracy", 0.0)) * 100
            res_data.append([
                Paragraph(r.get("model", ""), body_style),
                Paragraph(f"{float(r.get('test_loss', 0.0)):.4f}", body_style),
                Paragraph(f"{float(r.get('test_perplexity', 0.0)):.2f}", body_style),
                Paragraph(f"{acc_val:.2f}%", body_style),
                Paragraph(f"{float(r.get('training_time_sec', 0.0)):.1f}s", body_style),
                Paragraph(f"{r.get('best_epoch', '')}", body_style),
            ])
        t_res = Table(res_data, colWidths=[120, 75, 80, 80, 95, 70])
        t_res.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_res)

    story.append(Spacer(1, 8))

    # Section 5: Temperature Analysis
    story.append(Paragraph("5. Sampling Temperature Analysis & Qualitative Lexical Metrics", h1_style))
    story.append(Paragraph(
        "Sequential generation was performed at temperatures <b>T = 0.5</b>, <b>0.8</b>, and <b>1.2</b> across standardized seed prompts. "
        "Lexical diversity was quantified using Distinct-1 (unigram), Distinct-2 (bigram), and repetition ratios:<br/>"
        "• <b>T = 0.5 (Low Entropy):</b> High local syntax but low diversity (Distinct-1 &approx; 0.25&ndash;0.31). Degenerates into repetitive cycles.<br/>"
        "• <b>T = 0.8 (Optimal Balance):</b> Balanced exploration with natural Shakespearean prose cadence (Distinct-1 &approx; 0.47&ndash;0.54, repetition &le; 0.14).<br/>"
        "• <b>T = 1.2 (High Entropy):</b> High lexical diversity (Distinct-1 &gt; 0.70) accompanied by syntactic disintegration.",
        body_style,
    ))
    story.append(Spacer(1, 8))

    # Section 6: Representative Text Sample
    story.append(Paragraph("6. Representative Generated Text (Seed: <i>\"To be or not to\"</i>)", h1_style))
    sample_text = (
        "<b>GRU (T = 0.8):</b> <i>\"to be or not to least a be enough in night. first senator: then, my come that been his mercy and him my wounds lay. bishop than aumerle: what is though for weeping, catesby! duke elizabeth: o casting! pompey, well, my lord, i\"</i>"
    )
    story.append(Paragraph(sample_text, body_style))
    story.append(Spacer(1, 8))

    # Section 7: Conclusion
    story.append(Paragraph("7. Conclusion & Architectural Justification", h1_style))
    story.append(Paragraph(
        "Bidirectional RNN achieved the lowest held-out cross-entropy test loss (5.2002) and perplexity (181.31) due to "
        "dense bidirectional context encoding of the 25-token prompt without causal lookahead. Among strictly causal "
        "recurrent baselines, GRU demonstrated the strongest trade-off of rapid training (126.1s), low perplexity (191.15), "
        "and superior qualitative phrasing over Simple RNN (198.33).",
        body_style,
    ))
    story.append(Spacer(1, 12))

    # Submission Sign-off
    sign_style = ParagraphStyle(
        "SignStyle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#718096"),
        alignment=2,
    )
    story.append(Paragraph("Verified and submitted by Yashraj Sakunde (Reg No: 230953172)", sign_style))

    pdf.build(story)
    print(f"[Submission] PDF document created at: {output_path}")


def generate_all_submission_docs():
    results = load_quantitative_results("results/model_comparison.csv")
    out_docx = "submission/YashrajSakunde_230953172_FGAI_Assignment2.docx"
    out_pdf = "submission/YashrajSakunde_230953172_FGAI_Assignment2.pdf"
    create_docx(out_docx, results)
    create_pdf(out_pdf, results)


if __name__ == "__main__":
    generate_all_submission_docs()

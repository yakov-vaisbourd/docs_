---
marp: true
theme: default
paginate: true
math: katex
style: |
  section {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    font-family: 'Aptos', 'Calibri', 'Segoe UI', Arial, sans-serif;
    font-size: 26px;
    color: #1a1a1a;
    padding: 50px 60px;
  }
  h1 {
    font-family: 'Aptos Display', 'Calibri', 'Segoe UI', Arial, sans-serif;
    color: #156082;
    font-size: 38px;
    border-bottom: 3px solid #E97132;
    padding-bottom: 8px;
    margin-bottom: 18px;
  }
  h2, h3 {
    font-family: 'Aptos Display', 'Calibri', 'Segoe UI', Arial, sans-serif;
    color: #0E2841;
  }
  strong { color: #0E2841; }
  li::marker { color: #E97132; }
  section.title-slide {
    justify-content: center;
    padding: 0;
    color: #fff;
    background: linear-gradient(rgba(10,20,35,0.55), rgba(10,20,35,0.8)), url('img/slide1-bg.jpg');
    background-size: cover;
    background-position: center;
  }
  section.title-slide .kicker {
    font-size: 24px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #fff;
    opacity: 0.85;
    margin-bottom: 140px;
  }
  section.title-slide h1 {
    all: unset;
    display: block;
    font-family: 'Aptos Display', 'Calibri', 'Segoe UI', Arial, sans-serif;
    font-size: 64px;
    font-weight: 800;
    color: #fff;
    text-align: center;
    line-height: 1.15;
    letter-spacing: 2px;
  }
  section.title-slide .byline {
    text-align: center;
    margin-top: 40px;
    font-size: 22px;
    color: #fff;
    opacity: 0.9;
  }
  .row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 24px;
    width: 100%;
  }
  .row img {
    max-height: 380px;
    max-width: 100%;
    object-fit: contain;
    flex: 1;
  }
  .row.three img { max-height: 260px; }
  .row.small { margin-top: 4px; }
  .row.small img { max-height: 240px; }
  .flow-steps {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 12px;
  }
  .flow-steps .step {
    flex: 1;
    background: #f4f8fb;
    border: 2px solid #156082;
    border-radius: 10px;
    padding: 12px 12px;
    text-align: center;
    font-size: 18px;
    color: #0E2841;
  }
  .flow-steps .arrow {
    font-size: 26px;
    color: #E97132;
    font-weight: 700;
  }
  .formula-box {
    text-align: center;
    font-size: 27px;
    font-weight: 700;
    color: #0E2841;
    background: #f4f8fb;
    border: 2px solid #156082;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 14px auto 18px;
    width: fit-content;
  }
  .frac {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    vertical-align: middle;
    margin: 0 8px;
    line-height: 1.15;
  }
  .frac .num {
    padding: 0 4px 3px;
    border-bottom: 2px solid currentColor;
  }
  .frac .den {
    padding: 3px 4px 0;
  }
  .attn-matrix-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 10px;
  }
  .attn-matrix {
    border-collapse: collapse;
    font-size: 17px;
  }
  .attn-matrix th {
    background: #156082;
    color: #fff;
    padding: 6px 16px;
    font-weight: 700;
  }
  .attn-matrix th.corner {
    background: #0E2841;
    font-size: 13px;
  }
  .attn-matrix td {
    border: 1px solid #cfd8e3;
    padding: 8px 16px;
    text-align: center;
  }
  .attn-matrix td.key-label {
    background: #0E2841;
    color: #fff;
    font-weight: 700;
  }
  .attn-matrix td.visible {
    background: #eef6fc;
    color: #0E2841;
    font-weight: 700;
  }
  .attn-matrix td.masked {
    background: #eceff3;
    color: #9aa5b1;
  }
  .attn-matrix.compact th,
  .attn-matrix.compact td {
    padding: 5px 12px;
    font-size: 15px;
  }
  .attn-matrix-caption {
    margin-top: 10px;
    font-size: 15px;
    color: #555;
    font-style: italic;
    text-align: center;
  }
  .accum-diagram {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    margin-top: 20px;
  }
  .acc-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }
  .acc-token {
    background: #0E2841;
    color: #fff;
    padding: 6px 16px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 18px;
  }
  .acc-v {
    background: #156082;
    color: #fff;
    padding: 10px 18px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 22px;
  }
  .acc-weight {
    color: #E97132;
    font-weight: 700;
    font-size: 16px;
  }
  .acc-arrow {
    font-size: 30px;
    color: #E97132;
    font-weight: 700;
  }
  .acc-sum {
    background: #fff6ef;
    border: 2px solid #E97132;
    color: #0E2841;
    border-radius: 10px;
    padding: 16px 22px;
    font-weight: 700;
    font-size: 22px;
    text-align: center;
  }
  .split-right {
    display: flex;
    gap: 28px;
    align-items: center;
    margin-top: 10px;
  }
  .split-right .text-col {
    flex: 1 1 40%;
    font-size: 21px;
  }
  .split-right .img-col {
    flex: 0 0 58%;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .split-right .img-col img {
    width: 100%;
    max-height: 430px;
    object-fit: contain;
  }
  .bpe-frames {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 8px;
  }
  .text-lenses {
    display: flex;
    flex-direction: row;
    align-items: stretch;
    gap: 18px;
    margin-top: 24px;
  }
  .text-lenses .frame {
    flex: 1 1 0;
    min-width: 0;
    font-size: 20px;
    padding: 18px 22px;
  }
  .frame {
    border: 2px solid #156082;
    border-radius: 10px;
    padding: 10px 18px;
    background: #f4f8fb;
    font-size: 22px;
  }
  .frame.counts { border-color: #E97132; background: #fff6ef; }
  .frame.step { border-color: #0E2841; background: #eef1f5; }
  .frame-label {
    font-weight: 700;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
    color: #156082;
  }
  .frame.counts .frame-label { color: #E97132; }
  .frame.step .frame-label { color: #0E2841; }
  .embed-grid-wrap {
    margin-top: 18px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .embed-grid {
    border-collapse: collapse;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 17px;
  }
  .embed-grid th {
    background: #156082;
    color: #fff;
    padding: 7px 14px;
  }
  .embed-grid td {
    border: 1px solid #cfd8e3;
    padding: 7px 14px;
    text-align: center;
    color: #333;
  }
  .embed-grid td:first-child {
    font-weight: 700;
    color: #0E2841;
    text-align: left;
    font-family: 'Aptos', 'Calibri', 'Segoe UI', Arial, sans-serif;
  }
  .embed-grid td:nth-child(2) {
    font-weight: 700;
    color: #E97132;
  }
  .embed-grid-caption {
    margin-top: 10px;
    font-size: 16px;
    color: #555;
    font-style: italic;
  }
  .col-caption {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    flex: 1;
  }
  .col-caption span {
    font-weight: 600;
    color: #0E2841;
  }
  .landscape-grid {
    display: flex;
    gap: 12px;
    margin-top: 16px;
  }
  .landscape-col {
    flex: 1;
    background: #f4f8fb;
    border: 2px solid #156082;
    border-top: 6px solid #156082;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 15px;
  }
  .landscape-col .cat-label {
    font-weight: 700;
    color: #156082;
    font-size: 15px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
  }
  .landscape-col .cat-sub {
    font-style: italic;
    color: #0E2841;
    font-size: 13px;
    margin-bottom: 6px;
  }
  .landscape-col ul {
    margin: 0;
    padding-left: 16px;
  }
  .landscape-col li {
    margin-bottom: 3px;
  }
  .landscape-note {
    margin-top: 16px;
    background: #fff6ef;
    border-left: 4px solid #E97132;
    padding: 10px 16px;
    font-size: 16px;
    color: #5a3b1e;
  }
  section.center-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    text-align: center;
  }
  section.center-content > *:not(h1) {
    margin-top: auto;
    margin-bottom: auto;
  }
  .decode-table-wrap {
    margin-top: 14px;
  }
  .decode-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 18px;
  }
  .decode-table th {
    background: #E97132;
    color: #fff;
    padding: 10px 14px;
    text-align: left;
  }
  .decode-table td {
    border: 1px solid #f0c9ac;
    padding: 10px 14px;
    vertical-align: top;
  }
  .decode-table tr:nth-child(even) td {
    background: #fff6ef;
  }
  .decode-table td:first-child {
    font-weight: 700;
    color: #0E2841;
    white-space: nowrap;
  }
  .decode-table .sub {
    display: block;
    margin-top: 4px;
    font-size: 15px;
    color: #5a3b1e;
  }
  .bottleneck-grid {
    display: flex;
    gap: 16px;
    margin-top: 14px;
  }
  .bottleneck-tile {
    flex: 1;
    background: #f4f8fb;
    border: 2px solid #156082;
    border-radius: 10px;
    padding: 16px 14px;
    text-align: center;
  }
  .bottleneck-icon { font-size: 34px; margin-bottom: 8px; }
  .bottleneck-title {
    font-weight: 700;
    color: #0E2841;
    font-size: 18px;
    margin-bottom: 6px;
  }
  .bottleneck-sub {
    font-size: 15px;
    color: #444;
  }
  .blur-diagram {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin-top: 26px;
  }
  .blur-inputs {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .blur-in {
    background: #eef1f5;
    border: 1px solid #cfd8e3;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 15px;
    color: #0E2841;
    font-style: italic;
  }
  .blur-arrow { font-size: 26px; color: #E97132; font-weight: 700; }
  .blur-head {
    background: #0E2841;
    color: #fff;
    border-radius: 8px;
    padding: 14px 20px;
    font-weight: 700;
    font-size: 18px;
    text-align: center;
  }
  .blur-output {
    background: #eceff3;
    border: 2px dashed #9aa5b1;
    color: #667;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 16px;
    font-style: italic;
    text-align: center;
    max-width: 180px;
  }
  .slice-table-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 12px;
  }
  .slice-table {
    border-collapse: collapse;
    font-size: 16px;
  }
  .slice-table th {
    background: #156082;
    color: #fff;
    padding: 8px 18px;
    font-weight: 700;
  }
  .slice-table th.corner { background: #0E2841; }
  .slice-table td {
    border: 1px solid #cfd8e3;
    padding: 8px 18px;
    text-align: center;
  }
  .slice-table td.word-label {
    background: #0E2841;
    color: #fff;
    font-weight: 700;
  }
  .slice-table td.slice-cell {
    background: #eef6fc;
    color: #0E2841;
    font-weight: 600;
  }
  .slice-table td.slice-ellipsis {
    color: #9aa5b1;
    font-weight: 700;
  }
  .slice-table.compact th,
  .slice-table.compact td {
    padding: 5px 16px;
    font-size: 15px;
  }
  .bank-diagram {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 10px;
  }
  .bank-word {
    background: #0E2841;
    color: #fff;
    font-weight: 700;
    font-size: 22px;
    border-radius: 8px;
    padding: 8px 26px;
  }
  .bank-branches {
    display: flex;
    gap: 14px;
    margin-top: 14px;
  }
  .bank-branches .frame {
    flex: 1;
    font-size: 15px;
    padding: 8px 12px;
  }
  .bank-concat {
    margin-top: 14px;
    background: #fff6ef;
    border: 2px solid #E97132;
    color: #0E2841;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 700;
    font-size: 17px;
    text-align: center;
  }
  .concat-compare {
    display: flex;
    gap: 18px;
    margin-top: 14px;
  }
  .concat-path {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    border-radius: 10px;
    padding: 10px 14px;
    border: 2px solid;
  }
  .concat-path.bad { border-color: #9aa5b1; background: #f4f6f8; }
  .concat-path.good { border-color: #156082; background: #eef6fc; }
  .concat-path-label {
    font-weight: 700;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #0E2841;
    margin-bottom: 2px;
  }
  .concat-vec {
    font-family: 'Consolas', 'Courier New', monospace;
    background: #fff;
    border: 1px solid #cfd8e3;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 15px;
    color: #0E2841;
  }
  .concat-op { font-weight: 700; color: #E97132; font-size: 16px; }
  .concat-arrow { font-size: 18px; color: #E97132; }
  .concat-result {
    font-family: 'Consolas', 'Courier New', monospace;
    font-weight: 700;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 14px;
    text-align: center;
  }
  .concat-result.bad-result { background: #eceff3; color: #667; }
  .concat-result.good-result { background: #156082; color: #fff; }
---

<!-- Slide number: 1 -->
<!-- _class: title-slide -->

<div class="kicker">Mapless Autonomous Driving</div>

# Large Language Models
# The Foundation

<div class="byline">Itay Krisher<br>Septmeber 2026</div>

---
<!-- Slide number: 2 -->
# Module 1 – The Raw Material
"Before these networks can talk to us, they first have to learn to read the internet, and they have to learn to understand our world."
-Andraj Karpathy

<!-- Notes:
"To understand an LLM, we have to start before the code or the neural network even exists. We have to look at its food supply: data.
When people say we 'train AI on the internet,' they make it sound like the model is reading clean library books. In reality, the internet is incredibly messy. We get our raw data from web crawlers—programs that automatically browse billions of web pages and copy everything they find. This raw material is called Crawler Data.
If you look at the left side of the slide, this is what a raw web crawl actually looks like. It is full of HTML tags, JavaScript code, advertisement banners, navigation menus, and repetitive footers. If we fed this raw sludge straight into a model, the AI would waste its massive computational brainpower learning how to write web styling code instead of learning human logic. We need a refinery. That is where curated datasets like Hugging Face's FineWeb come in."

When you present Slide 2, emphasize to your team that "garbage in, garbage out" is the golden rule of AI. Make it clear that before we ever talk about artificial intelligence, attention mechanisms, or complex math layers, thousands of hours are spent simply cleaning up the messy text left behind by internet users.
-->

---
<!-- Slide number: 3 -->
# Demystifying Raw Crawler Data

<div class="split-right">
  <div class="text-col">
    <p><strong>Raw Crawler Data</strong><br>
    Automate a web crawler (like Common Crawl) to scrape billions of web pages. They harvest the literal, unformatted code blocks from internet servers.</p>
    <p><strong>The Complexity Dilemma</strong><br>
    Models trained directly on raw, unfiltered code files waste valuable neural capacity learning trivial formatting rules rather than logical semantics.<br>
    How do we aggressively filter the data without losing valuable knowledge?</p>
  </div>
  <div class="img-col">
    <img src="img/slide3-1.png" alt="Raw crawled scraping sludge example">
  </div>
</div>

<!-- Notes:
"To understand an LLM, we have to start before the code or the neural network even exists. We have to look at its food supply: data.
When people say we 'train AI on the internet,' they make it sound like the model is reading clean library books. In reality, the internet is incredibly messy. We get our raw data from web crawlers—programs that automatically browse billions of web pages and copy everything they find. This raw material is called Crawler Data.
If you look at the left side of the slide, this is what a raw web crawl actually looks like. It is full of HTML tags, JavaScript code, advertisement banners, navigation menus, and repetitive footers. If we fed this raw sludge straight into a model, the AI would waste its massive computational brainpower learning how to write web styling code instead of learning human logic. We need a refinery. That is where curated datasets like Hugging Face's FineWeb come in."

When you present Slide 2, emphasize to your team that "garbage in, garbage out" is the golden rule of AI. Make it clear that before we ever talk about artificial intelligence, attention mechanisms, or complex math layers, thousands of hours are spent simply cleaning up the messy text left behind by internet users.
-->

---
<!-- Slide number: 4 -->
# Inside the FineWeb Refinery Pipeline

<div class="row">
  <img src="img/slide4-1.png" alt="FineWeb refinery pipeline: text extraction, quality filters, language filter, deduplication and PII">
</div>

---
<!-- Slide number: 5 -->
# The Power of Strategic Data Compression

<div class="row">
  <img src="img/slide5-1.png" alt="44 TB pristine refined text compression summary">
</div>

<!-- Notes:
"The mind-blowing takeaway of this curation process is the sheer scale of compression. The internet feels infinite to us. If you include all the videos, raw code, and media, we are talking about petabytes of data.
But once you aggressively filter for just high-quality, unique human thought and text, the entire curated dataset drops down to roughly 44 Terabytes.
Think about that perspective: almost the entirety of public human knowledge, history, science, and literature on the internet can be compressed small enough to sit on a single hard drive that you can buy at a local electronics store. This clean 44-Terabyte text file is the exact foundation we hand over to the Tokenizer, which is what we will explore next."
-->

---
<!-- Slide number: 6 -->
# The Dataset Landscape
Common Crawl and FineWeb cover pretraining — but building and evaluating an LLM draws on a much wider ecosystem of data:

<div class="landscape-grid">
  <div class="landscape-col">
    <div class="cat-label">Pretraining</div>
    <div class="cat-sub">Raw text &rarr; learns language</div>
    <ul>
      <li>Common Crawl</li>
      <li>FineWeb / FineWeb-Edu</li>
      <li>The Pile</li>
      <li>RedPajama</li>
    </ul>
  </div>
  <div class="landscape-col">
    <div class="cat-label">Instruction-Tuning</div>
    <div class="cat-sub">Teaches it to follow commands</div>
    <ul>
      <li>Alpaca</li>
      <li>OpenAssistant (OASST)</li>
      <li>Dolly 15k</li>
    </ul>
  </div>
  <div class="landscape-col">
    <div class="cat-label">Preference / RLHF</div>
    <div class="cat-sub">Polishes responses via human feedback</div>
    <ul>
      <li>Anthropic HH-RLHF</li>
      <li>OpenAI WebGPT</li>
      <li>Stanford SHP</li>
    </ul>
  </div>
  <div class="landscape-col">
    <div class="cat-label">Evaluation</div>
    <div class="cat-sub">Measures how well it worked</div>
    <ul>
      <li>MMLU</li>
      <li>HellaSwag</li>
      <li>GSM8K / MATH</li>
      <li>HumanEval</li>
    </ul>
  </div>
  <div class="landscape-col">
    <div class="cat-label">Multimodal</div>
    <div class="cat-sub">Extends training/eval to images &amp; documents</div>
    <ul>
      <li>LAION-5B</li>
      <li>COCO</li>
      <li>DocVQA</li>
    </ul>
  </div>
</div>

<div class="landscape-note">Many raw web-scraped sets carry copyright/licensing disputes, and benchmark contamination — test data leaking into training — remains a persistent risk. New datasets and benchmarks appear monthly.</div>

<!-- Notes:
This is a bridge slide: Module 1 walked through one pipeline (Common Crawl to FineWeb) for pretraining text, but that's one slice of a much bigger picture.
Pretraining, Instruction-Tuning, and Preference/RLHF are genuine sequential stages — each one takes the model produced by the previous stage and refines its behavior further:
- Pretraining teaches raw language ability from massive text corpora (Common Crawl, FineWeb, The Pile, RedPajama).
- Instruction-tuning retrains the pretrained model on (instruction, ideal response) pairs so it behaves like an assistant rather than just continuing text (Alpaca, OASST, Dolly 15k).
- Preference/RLHF further fine-tunes it using human rankings of which responses are better, which is what gives modern assistants their polished tone (Anthropic HH-RLHF, WebGPT, SHP).
Evaluation and Multimodal are NOT further stages in that pipeline — flag this distinction explicitly:
- Evaluation benchmarks are tests run after (and during) training to measure how well it worked — knowledge (MMLU), reasoning (ARC, BIG-bench), math (GSM8K/MATH), code (HumanEval, SWE-bench), truthfulness (TruthfulQA).
- Multimodal datasets (LAION-5B, COCO, DocVQA) extend any of the above stages to images and documents, not just text.
Worth flagging to the audience: copyright/licensing disputes over scraped data (Common Crawl, LAION, Books3), and benchmark contamination (test sets leaking into pretraining corpora), are two of the biggest open problems in this space right now.
-->

---
<!-- Slide number: 7 -->
# **What is a text?**

<div class="text-lenses">
  <div class="frame corpus">
    <div class="frame-label">The Human &amp; Communication Lens</div>
    Text is a tool for <strong>transmitting thought across space and time</strong>. Unlike spoken conversation, which is immediate and temporary, text fixes a message into a stable form. It allows a person today to read the exact thoughts written by an author thousands of years ago, or an engineer in Asia to read instructions written by a developer in Europe.
  </div>
  <div class="frame counts">
    <div class="frame-label">The Data &amp; Computer Science Lens</div>
    To a computer or an AI model, text is <strong>a structured array of digital characters encoded into binary numbers</strong>.
  </div>
</div>

<!-- Notes:
Use this slide to establish the two meanings of text before introducing tokenization. For humans, text preserves and transmits thought across distance and time. For computers, text is structured digital data: characters represented numerically, ultimately as binary. The next slide explains how tokenization turns that data into chunks an LLM can process.
-->

---
<!-- Slide number: 7 -->
# Module 2: The Translation Layer — What is a Token?
How LLMs turn raw characters into chunks they can compute, and the cognitive costs of doing so.

<div class="row">
  <img src="img/slide6-1.png" alt="A pile of word tiles">
  <img src="img/slide6-2.png" alt="Binary digits background">
</div>

<!-- Notes:
State the core transition: In Module 1, we learned how we get a clean 44-Terabyte text file. But computers cannot run math on raw letters like 'c-a-t'. They only compute integers. This module covers the translation layer that bridges human language and machine computation.
-->

---
<!-- Slide number: 8 -->
# How Do Computers Read? The Core Paradox

| | Approach 1: Character-by-Character<br>('c', 'a', 't') | Approach 2: Whole-Word Dictionary<br>('cat' = 12, 'dog' = 13) |
|---|---|---|
| **Vocabulary size** | Tiny (~256 symbols) | Explodes into millions of entries |
| **Sequence length** | Blows up — too many tokens per sentence | Short and dense |
| **Handles typos / new words / code?** | Yes, but wastes compute just learning how to spell | No — breaks instantly on out-of-vocabulary text |

Neither extreme works well — we need a middle ground: **subwords**.

<!-- Notes:
Explain the tension between characters and full words.
If we use letters, the input sequence is way too long and computationally heavy. If we use full words, the dictionary is too rigid and completely breaks if someone makes a typo or writes a line of code.
We need a compromise that sits right in the middle: subwords.
-->

---
<!-- Slide number: 9 -->
# Byte Pair Encoding (BPE)

<div class="bpe-frames">
  <div class="frame corpus">
    <div class="frame-label">What is BPE?</div>
    A text compression algorithm that finds a middle ground by splitting language into common subword chunks.
  </div>
  <div class="frame counts">
    <div class="frame-label">How it builds its vocabulary</div>
    <div class="flow-steps">
      <div class="step">Start: every character is its own token</div>
      <div class="arrow">&rarr;</div>
      <div class="step">Count the most frequent adjacent pair<br>(e.g. 't' + 'h')</div>
      <div class="arrow">&rarr;</div>
      <div class="step">Merge it into a new token<br>(e.g. 'th')</div>
      <div class="arrow">&rarr;</div>
      <div class="step">Repeat until target vocab size is reached<br>(e.g. 50,000 IDs)</div>
    </div>
  </div>
  <div class="frame step">
    <div class="frame-label">Visualizing the subword output</div>
    'strawberry' &rarr; [straw] [berry] (2 tokens instead of 10 characters)<br>
    'tokenization' &rarr; [token] [iz] [ation] (3 tokens instead of 12 characters)
  </div>
</div>

<!-- Notes:
TALKING POINTS:
Explain how BPE is constructed. It's a simple merge loop.
Common things like 'the' or 'and' get compressed into single standalone tokens. Rare words get broken down into pieces based on their frequency.
Point out the 'strawberry' example. It's split into 'straw' and 'berry'—this is fundamental to explaining the next slide.
-->

---
<!-- Slide number: 10 -->
# BPE Example

<div class="bpe-frames">
  <div class="frame corpus">
    <div class="frame-label">Corpus (dataset)</div>
    low, low, low, low, low, lower, lower, lower, newest, newest, newest, newest, newest, newest, newest, widest, widest, widest
  </div>
  <div class="frame counts">
    <div class="frame-label">Word counts</div>
    low (5) &nbsp;&middot;&nbsp; lower (3) &nbsp;&middot;&nbsp; newest (6) &nbsp;&middot;&nbsp; widest (3)
  </div>
  <div class="frame step">
    <div class="frame-label">Step 0 &mdash; Vocabulary</div>
    Vocab = {'l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'd', 'i'}
  </div>
</div>

---
<!-- Slide number: 11 -->
# BPE Example

<div class="bpe-frames">
  <div class="frame corpus">
    <div class="frame-label">Split the words</div>
    l o w (5) &nbsp;&middot;&nbsp; l o w e r (3) &nbsp;&middot;&nbsp; n e w e s t (6) &nbsp;&middot;&nbsp; w i d e s t (3)
  </div>
  <div class="frame counts">
    <div class="frame-label">Iteration 1 &mdash; merge step</div>
    Scans the text and counts which two characters appear next to each other most often.<br>'e' and 's' appear 9 times.
  </div>
  <div class="frame step">
    <div class="frame-label">Step 1 &mdash; Vocabulary</div>
    Vocab = {'l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'd', 'i', <strong>'es'</strong>}
  </div>
</div>

---
<!-- Slide number: 12 -->
# BPE Example

<div class="bpe-frames">
  <div class="frame corpus">
    <div class="frame-label">New set</div>
    l o w (5) &nbsp;&middot;&nbsp; l o w e r (3) &nbsp;&middot;&nbsp; n e w <strong>es</strong> t (6) &nbsp;&middot;&nbsp; w i d <strong>es</strong> t (3)
  </div>
  <div class="frame counts">
    <div class="frame-label">Iteration 2 &mdash; merge step</div>
    Scans the text and counts which two pairs appear next to each other most often.<br>'es' and 't' appear 9 times.
  </div>
  <div class="frame step">
    <div class="frame-label">Step 2 &mdash; Vocabulary</div>
    Vocab = {'l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'd', 'i', 'es', <strong>'est'</strong>}
  </div>
</div>

---
<!-- Slide number: 13 -->
# BPE Example

<div class="bpe-frames">
  <div class="frame corpus">
    <div class="frame-label">New set</div>
    l o w (5) &nbsp;&middot;&nbsp; l o w e r (3) &nbsp;&middot;&nbsp; n e w <strong>est</strong> (6) &nbsp;&middot;&nbsp; w i d <strong>est</strong> (3)
  </div>
  <div class="frame counts">
    <div class="frame-label">Iteration 3 &mdash; merge step</div>
    'l' and 'o' appear together most often and are merged into a new token: 'lo'.
  </div>
  <div class="frame step">
    <div class="frame-label">After iteration 3 &mdash; Vocabulary</div>
    Vocab = {'l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'd', 'i', 'es', 'est', <strong>'lo'</strong>}
  </div>
</div>

<!-- Notes:
Analogy - LEGO
-->

---
<!-- Slide number: 14 -->
# BPE Flow

<div class="row">
  <img src="img/slide14-1.png" alt="BPE vocabulary-building flow diagram">
</div>

---
<!-- Slide number: 15 -->
# Chat Template: Hidden Conversation Markers

<div class="row">
  <div class="frame">
    <div class="frame-label">A structured message becomes a token sequence</div>
    <code>&lt;|im_start|&gt;system&lt;|im_sep|&gt;You are helpful.&lt;|im_end|&gt;</code>
  </div>
</div>

- <strong><code>&lt;|im_start|&gt;</code></strong> — marks the beginning of a chat message.
- **`system`** — identifies the message's role: instructions for the assistant.
- <strong><code>&lt;|im_sep|&gt;</code></strong> — separates the role from the message content.
- **`You are helpful.`** — the actual content of the system message.
- <strong><code>&lt;|im_end|&gt;</code></strong> — marks the end of the message.

<div class="formula-box">
  Messages &rarr; chat template &rarr; formatted text + control tokens &rarr; tokenizer &rarr; token IDs
</div>

<small><strong>Note:</strong> Names and behavior vary by model. The model must be trained with the same template it receives at inference.</small>

<!-- Notes:
The exact markers are model-specific. <code>system</code> identifies the role; the start, separator, and end markers tell the model where a message begins, what role/content follows, and where it ends. A user usually sends structured messages such as {role: "system", content: "You are helpful."}; the software formats them before tokenization.
-->

---
<!-- Slide number: 16 -->
# Tokenizer Example
[Tiktokenizer](https://tiktokenizer.vercel.app/)

---
<!-- Slide number: 17 -->
# Module 3: Injected Meaning
So far, we have translated words and letters to tokens.
Tokens are just numbers. They don't have a meaning.
**Q:** How do we get from tokens to meanings?
**A:** Embeddings & Positional Encoding

---
<!-- Slide number: 18 -->
# Token Embeddings: Mapping Meaning
How do LLM's understand what words mean?
Example: What does the word 'Train' mean?
- I train my dog.
- The big yellow train leaves in one hour.

LLM needs to be taught context.

---
<!-- Slide number: 19 -->
# Visualization of Embedding

<div class="row">
  <img src="img/slide18-1.png" alt="Words mapped through embedding into vectors in vector space">
</div>

---
<!-- Slide number: 20 -->
# Vector Embedding
- Every unique token in the vocabulary gets its own row in a lookup table.
- Vocab of 50,000 tokens × 4,096 dimensions &rarr; a 50,000 x 4,096 grid of numbers.
- Training starts with the table filled with random decimals — the model has no idea what "cat" or "mouse" means yet.

<div class="embed-grid-wrap">
  <table class="embed-grid">
    <thead>
      <tr><th>word</th><th>token id</th><th>d0</th><th>d1</th><th>d2</th><th>d3</th><th>&hellip;</th><th>d4095</th></tr>
    </thead>
    <tbody>
      <tr><td>cat</td><td>1723</td><td>0.42</td><td>-1.07</td><td>0.88</td><td>-0.15</td><td>&hellip;</td><td>0.31</td></tr>
      <tr><td>dog</td><td>4091</td><td>-0.63</td><td>0.29</td><td>-1.42</td><td>0.77</td><td>&hellip;</td><td>-0.94</td></tr>
      <tr><td>mouse</td><td>9852</td><td>1.18</td><td>-0.05</td><td>0.36</td><td>-1.21</td><td>&hellip;</td><td>0.62</td></tr>
      <tr><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td></tr>
    </tbody>
  </table>
  <div class="embed-grid-caption">50,000 rows &times; 4,096 columns — every cell starts as a random decimal</div>
</div>

---
<!-- Slide number: 20 -->
<!-- _class: center-content -->
# Attention Layer

---
<!-- Slide number: 21 -->
# The Communication Layer — The Self-Attention Mechanism
**The Bottleneck of Static Representations**
- Static token embeddings act like dictionary lookups and page numbers.
- On their own, they are completely rigid.
- The vector for a word is identical regardless of whether it's next to its neighbors.

**"Tower" — which one did you mean?**

<div class="row three">
  <div class="col-caption"><img src="img/slide26-1.png" alt="Eiffel Tower"><span>Eiffel Tower</span></div>
  <div class="col-caption"><img src="img/slide26-4.png" alt="Tower Bridge"><span>Tower Bridge</span></div>
  <div class="col-caption"><img src="img/slide26-3.png" alt="Computer Tower"><span>Computer Tower</span></div>
</div>

---
<!-- Slide number: 22 -->
# The Self-Attention Mechanism

<div class="row">
  <img src="img/slide27-1.png" alt="Self-attention visualization showing tokens attending to each other with weighted lines">
</div>

As the model "reads" a sequence, the self-attention layer allows every token to examine all other tokens simultaneously.
Tokens exchange information and update their vector representations in real time based on their surrounding context.

---
<!-- Slide number: 23 -->
# The Matchmakers — Queries, Keys, and Values (Q, K, V)
- **Q** – Query (Vector)
- **K** – Key (Vector)
- **V** – Value (Vector)

Each token is projected through learned weight matrices (Wq, Wk, Wv) into its Query, Key, and Value vectors. The dot product of Queries and Keys determines how much attention one token pays to another.

<div class="row small">
  <img src="img/slide28-1.png" alt="Attention weight matrix w sub ij">
</div>

---
<!-- Slide number: 24 -->
# Step 1 — The Dot Product & the Attention Pattern

<!-- <div class="formula-box">Attention(Q, K, V) = softmax( <span class="frac"><span class="num">QK<sup>T</sup></span><span class="den">&radic;d<sub>k</sub></span></span> ) V</div> -->

<!-- <div class="formula-box">  -->
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}V\right)
$$
<!-- </div> -->



Every token's Query is compared against every other token's Key with a dot product: **score<sub>ij</sub> = Q<sub>i</sub> &middot; K<sub>j</sub>**. For a 4-token sentence, that gives a 4&times;4 grid of raw scores — the **attention pattern**.

<div class="attn-matrix-wrap">
  <table class="attn-matrix">
    <thead>
      <tr><th class="corner">Key &darr; / Query &rarr;</th><th>The</th><th>cat</th><th>sat</th><th>down</th></tr>
    </thead>
    <tbody>
      <tr><td class="key-label">The</td><td class="visible">s<sub>1,1</sub></td><td class="visible">s<sub>1,2</sub></td><td class="visible">s<sub>1,3</sub></td><td class="visible">s<sub>1,4</sub></td></tr>
      <tr><td class="key-label">cat</td><td class="masked">0</td><td class="visible">s<sub>2,2</sub></td><td class="visible">s<sub>2,3</sub></td><td class="visible">s<sub>2,4</sub></td></tr>
      <tr><td class="key-label">sat</td><td class="masked">0</td><td class="masked">0</td><td class="visible">s<sub>3,3</sub></td><td class="visible">s<sub>3,4</sub></td></tr>
      <tr><td class="key-label">down</td><td class="masked">0</td><td class="masked">0</td><td class="masked">0</td><td class="visible">s<sub>4,4</sub></td></tr>
    </tbody>
  </table>
  <div class="attn-matrix-caption">Rows = Key (the word being looked at) &middot; Columns = Query (the word asking the question) &middot; s<sub>i,j</sub> = raw score between Key i and Query j — each one a <em>different</em> number, not literally equal</div>
</div>

<div class="landscape-note"><strong>Why the lower triangle is zero:</strong> a Query can only attend to Keys at or before its own position. Whenever the Key sits <em>after</em> the Query in the sentence, that cell falls below the diagonal in this grid — the model forces that score to &minus;&infin; before Softmax, so it becomes exactly 0. This causal mask is what stops the model from "reading ahead" during training.</div>

<!-- Notes:
score_ij = Qi · Kj. Walk the grid: the diagonal is always visible (a token can always attend to itself), everything above-right of the diagonal is visible (keys at-or-before the query), everything below-left is masked to zero (keys that come after the query — the future).
-->

---
<!-- Slide number: 25 -->
# Step 2 — Scaling by &radic;d<sub>k</sub> & Softmax

$$
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

<div style="display:flex; gap:14px; margin-top:2px;">
  <div class="frame counts" style="flex:1; font-size:18px; padding:8px 14px;">
    <div class="frame-label">Why divide by &radic;d<sub>k</sub>?</div>
    Dot-product variance grows with d<sub>k</sub>. Unscaled, scores blow up and saturate Softmax (near-0/1, vanishing gradient) — dividing by &radic;d<sub>k</sub> keeps them stable.
  </div>
  <div class="frame step" style="flex:1; font-size:18px; padding:8px 14px;">
    <div class="frame-label">Softmax</div>
    Turns each Query's column of scaled scores into a probability distribution that sums to 1 — masked (future) entries stay exactly 0.
  </div>
</div>

<div class="attn-matrix-wrap">
  <table class="attn-matrix compact">
    <thead>
      <tr><th class="corner">Key &darr; / Query &rarr;</th><th>The</th><th>cat</th><th>sat</th><th>down</th></tr>
    </thead>
    <tbody>
      <tr><td class="key-label">The</td><td class="visible">1.00</td><td class="visible">0.35</td><td class="visible">0.20</td><td class="visible">0.10</td></tr>
      <tr><td class="key-label">cat</td><td class="masked">0</td><td class="visible">0.65</td><td class="visible">0.30</td><td class="visible">0.15</td></tr>
      <tr><td class="key-label">sat</td><td class="masked">0</td><td class="masked">0</td><td class="visible">0.50</td><td class="visible">0.25</td></tr>
      <tr><td class="key-label">down</td><td class="masked">0</td><td class="masked">0</td><td class="masked">0</td><td class="visible">0.50</td></tr>
    </tbody>
  </table>
  <div class="attn-matrix-caption">Each column (one Query) sums to 1 — these are the final attention weights &beta;<sub>ij</sub></div>
</div>

<!-- Notes:
What is dk? It's the dimensionality of the Query/Key vectors that Wq/Wk produce. We're still single-head here, so dk is just the model's embedding dimension — e.g. dk = 4,096 (not split across heads yet — that's the next slide's job).
Each column is one query token's finished attention distribution — e.g. "sat" (3rd column) puts 20% of its attention on "The", 30% on "cat", 50% on itself, 0% on "down" (future, masked).
Deliberately single-head here: dk is the full embedding dimension, not a per-head slice. Splitting dk across heads is introduced on the next slide (Multi-Head Attention) — don't front-run that concept here.
-->

---
<!-- Slide number: 26 -->
# Step 3 — Accumulating Values into &Delta;E

Building on the Q/K/V projection from the Matchmakers slide: once every token has a Value vector and the Query "sat" has its attention weights &beta;<sub>ij</sub> from Step 2, it blends in a weighted sum of the Values it's allowed to see.

<div class="accum-diagram">
  <div class="acc-col"><div class="acc-token">The</div><div class="acc-v">V<sub>1</sub></div><div class="acc-weight">&beta; = 0.20</div></div>
  <div class="acc-col"><div class="acc-token">cat</div><div class="acc-v">V<sub>2</sub></div><div class="acc-weight">&beta; = 0.30</div></div>
  <div class="acc-col"><div class="acc-token">sat</div><div class="acc-v">V<sub>3</sub></div><div class="acc-weight">&beta; = 0.50</div></div>
  <div class="acc-arrow">&rarr;</div>
  <div class="acc-sum">&Sigma; &beta;<sub>j</sub>V<sub>j</sub><br>= &Delta;E</div>
</div>

<div class="formula-box" style="margin-top:22px;">E&prime;<sub>sat</sub> = E<sub>sat</sub> + &Delta;E</div>

<div class="landscape-note">"down" contributes <strong>nothing</strong> to this update — its weight was masked to 0 in Step 2, so its Value is excluded entirely. This is why attention is often described as "moving information around": relevant Values get blended into the Query's own vector, while masked future tokens contribute exactly zero.</div>

<!-- Notes:
This is the ΔE from the formula's final "×V" step, made concrete: it's a weighted sum of Value vectors, weighted by the Softmax probabilities computed in Step 2, restricted to the tokens the causal mask allows.
-->

---
<!-- Slide number: 27 -->
<!-- _class: center-content -->
# Multi-Head Attention

---
<!-- Slide number: 28 -->
# The Bottleneck — Why One Head Isn't Enough

<div class="landscape-note" style="margin-top:0;">"The bank was closed during the holiday."</div>

<div class="bottleneck-grid">
  <div class="bottleneck-tile">
    <div class="bottleneck-icon">🔀</div>
    <div class="bottleneck-title">Single-Focus Bottleneck</div>
    <div class="bottleneck-sub">One relationship at a time — forced to pick just the dominant one</div>
  </div>
  <div class="bottleneck-tile">
    <div class="bottleneck-icon">🌫️</div>
    <div class="bottleneck-title">Information Smearing</div>
    <div class="bottleneck-sub">Softmax averages distant signals into one blurry blend</div>
  </div>
  <div class="bottleneck-tile">
    <div class="bottleneck-icon">📚</div>
    <div class="bottleneck-title">Capacity Constraints</div>
    <div class="bottleneck-sub">Grammar, semantics, structure — all competing for one lens</div>
  </div>
</div>

<div class="blur-diagram">
  <div class="blur-inputs">
    <div class="blur-in">"closed" + "holiday"</div>
    <div class="blur-in">"The"</div>
    <div class="blur-in">"was closed"</div>
  </div>
  <div class="blur-arrow">&rarr;</div>
  <div class="blur-head">1 Head</div>
  <div class="blur-arrow">&rarr;</div>
  <div class="blur-output">one blurred, averaged signal</div>
</div>

<!-- Notes:
Sets up the contrast for the next two slides: a single head can't separate what "bank" means, how it functions grammatically, and what it refers to — it has to blend all three into one averaged signal.
-->

---
<!-- Slide number: 29 -->
# Splitting the Depth, Not the Sentence

<div class="landscape-note" style="margin-top:0;">"The bank was closed during the holiday." — every head sees the <strong>entire sentence</strong>; what's split is the embedding's internal depth, not the words.</div>

<div class="slice-table-wrap">
  <table class="slice-table compact">
    <thead>
      <tr><th class="corner"></th><th>Head 1</th><th>Head 2</th><th>&hellip;</th><th>Head 8</th></tr>
    </thead>
    <tbody>
      <tr><td class="word-label">The</td><td class="slice-cell">First 64</td><td class="slice-cell">Next 64</td><td class="slice-ellipsis">&hellip;</td><td class="slice-cell">Last 64</td></tr>
      <tr><td class="word-label">bank</td><td class="slice-cell">First 64</td><td class="slice-cell">Next 64</td><td class="slice-ellipsis">&hellip;</td><td class="slice-cell">Last 64</td></tr>
      <tr><td class="word-label">was</td><td class="slice-cell">First 64</td><td class="slice-cell">Next 64</td><td class="slice-ellipsis">&hellip;</td><td class="slice-cell">Last 64</td></tr>
      <tr><td class="word-label">closed</td><td class="slice-cell">First 64</td><td class="slice-cell">Next 64</td><td class="slice-ellipsis">&hellip;</td><td class="slice-cell">Last 64</td></tr>
      <tr><td class="word-label">during</td><td class="slice-cell">First 64</td><td class="slice-cell">Next 64</td><td class="slice-ellipsis">&hellip;</td><td class="slice-cell">Last 64</td></tr>
      <tr><td class="word-label">the</td><td class="slice-cell">First 64</td><td class="slice-cell">Next 64</td><td class="slice-ellipsis">&hellip;</td><td class="slice-cell">Last 64</td></tr>
      <tr><td class="word-label">holiday</td><td class="slice-cell">First 64</td><td class="slice-cell">Next 64</td><td class="slice-ellipsis">&hellip;</td><td class="slice-cell">Last 64</td></tr>
    </tbody>
  </table>
  <div class="attn-matrix-caption">Same sentence, different lens — each head only sees its own 64-number slice</div>
</div>

<div class="formula-box" style="font-size:20px; padding:10px 16px;">d<sub>model</sub> 512 &divide; 8 heads = d<sub>k</sub> 64 per head</div>

<!-- Notes:
Common misconception: the sentence is NOT divided among heads. Every head processes all 5 words — only the 512-number embedding per word gets sliced into 8 chunks of 64.
-->

---
<!-- Slide number: 30 -->
# Three Lenses on One Word — "bank"

<div class="landscape-note" style="margin-top:0;">"The <strong>bank</strong> was closed during the <strong>holiday</strong>."</div>

<div class="bank-diagram">
  <div class="bank-word">bank</div>
  <div class="bank-branches">
    <div class="frame counts">
      <div class="frame-label">Head 1 — Semantic</div>
      attends to "closed", "holiday" &rarr; financial institution
    </div>
    <div class="frame step">
      <div class="frame-label">Head 2 — Grammar</div>
      attends to "The" &rarr; primary noun / subject
    </div>
    <div class="frame">
      <div class="frame-label">Head 3 — Predicate Link</div>
      attends to "was closed" &rarr; connects "bank" to its state
    </div>
  </div>
</div>

<div class="concat-compare">
  <div class="concat-path bad">
    <div class="concat-path-label">Sum / Average</div>
    <div class="concat-vec">[10, -5]</div>
    <span class="concat-op">+</span>
    <div class="concat-vec">[-10, 5]</div>
    <div class="concat-arrow">&darr;</div>
    <div class="concat-result bad-result">[0, 0] — destroyed</div>
  </div>
  <div class="concat-path good">
    <div class="concat-path-label">Concatenate</div>
    <div class="concat-vec">[10, -5]</div>
    <span class="concat-op">+</span>
    <div class="concat-vec">[-10, 5]</div>
    <div class="concat-arrow">&darr;</div>
    <div class="concat-result good-result">[10, -5, -10, 5] — both preserved</div>
  </div>
</div>

<div class="formula-box" style="margin-top:10px; font-size:19px; padding:8px 14px;">MultiHead(Q,K,V) = Concat(head<sub>1</sub>, &hellip;, head<sub>h</sub>)W<sup>O</sup></div>

<!-- Notes:
Contrast with the bottleneck slide: three independent heads compute in parallel — Head 1 ("financial institution") ≈ [10,-5], Head 2 ("is a noun") ≈ [-10,5]. Summing/averaging cancels them toward zero; concatenation keeps both signals fully intact side-by-side, which is what makes the parallel heads' work actually usable downstream.
-->

---
<!-- Slide number: 31 -->
# Module 4 - Positional Encoding
In older AI models (like RNNs or LSTMs), text was processed sequentially, word by word, from left to right.
Without intervention, these sentences have the same meaning:
- The cat ate the mouse
- The mouse ate the cat

To solve the word order problem, we use Positional Encoding.
There are two methods of Positional Encoding:
- Absolute Positional Encodings
- Rotary Position Embeddings (RoPE)

---
<!-- Slide number: 32 -->
# Absolute Positional Encodings
Give every position a unique signature without forcing the model to learn extra parameters during training.

<div class="row">
  <img src="img/slide21-1.png" alt="Positional encoding sine formula">
  <img src="img/slide21-2.png" alt="Positional encoding cosine formula">
</div>

i = index from 0 to dmodel/2-1
pos = position of word in the sentence
dmodel = dimension of word vector

<!-- Notes:
The Core Concept
Imagine a soundboard with hundreds of sliders, each oscillating at a different frequency.
Position 1 gets a specific snapshot of all those waves.
Position 2 gets a slightly advanced snapshot of those same waves.
Because sine and cosine functions repeat in a predictable, smooth pattern between -1 and 1, the network doesn't just learn "I am at index 5." It learns to recognize the geometric vibe of being at index 5 based on the overlapping wave patterns.
The sin and cos functions let us get values like a different clock for each slot of the array.
-->

---
<!-- Slide number: 33 -->
# Absolute Positional Encodings – calculation example
Let's consider the word 'cat' from the sentence 'The cat ate'.
pos = 1; i runs from 0 to 2047; dmodel = 4096

| Slot pair | i | Slot 0 PE | Slot 1 PE |
|---|---|---|---|
| Slots 0 and 1 | 0 | 0.175 | 0.999 |
| Slots 2 and 3 | 1 | 8.522e-6 | 0.9999999999… |

---
<!-- Slide number: 34 -->
# The Entire process with Positional Encoding

<div class="row">
  <img src="img/slide23-1.png" alt="Full process: vocab, embedding dictionary, positional encoding, final network input">
</div>

---
<!-- Slide number: 35 -->
# Positional Encoding – Limitations
Limited by the historical horizon of the training data.
Absolute positional encoding acts like a rigid map: if the model has never walked past mile marker 2,048 during its education, it gets completely lost the moment you take it to mile marker 2,049

<div class="row">
  <img src="img/slide24-1.png" alt="Musical staff notes within trained range">
  <img src="img/slide24-2.png" alt="Musical staff notes highlighting out-of-range note">
</div>

<!-- Notes:
The Limitation
Because these are absolute coordinates assigned linearly, the model struggles to generalize to text sequences longer than what it saw during training. If it only trained on sequences up to 2,048 tokens, it has no idea what the "wave signature" of token 4,096 should mean.
The Extrapolation Problem (Out of Bounds)
Imagine you are teaching a child to read music, but you only ever show them notes on a standard 5-line staff. They learn exactly what every note means within that boundary.
If you suddenly hand them a piece of music with extra lines drawn way above the staff (ledger lines) representing incredibly high pitches, they will freeze. Even though the mathematical logic of the music scale is exactly the same up there, they have never seen notes in that visual territory before, so they have no pattern to match it to.
-->

---
<!-- Slide number: 36 -->
# Module 5: Detokenization — From Math Back to Words
Detokenization is the final assembly line of LLMs — turning a series of math IDs back into human words, one piece at a time.

<!-- Notes:
Bridge slide. Everything up to now has been the forward pass: text in, tokens in, the network reasons over embeddings, attention, and position. Now we close the loop — how does the network's raw output actually become the words we read on screen?
-->

---
<!-- Slide number: 37 -->
# Step 1: Decoding — Choosing the Next Word Piece
The model doesn't generate entire sentences at once.
It predicts **one token (word piece) at a time** — scoring thousands of vocabulary candidates with a probability distribution — then a **decoding strategy** picks the winner.

<div class="flow-steps">
  <div class="step">Network outputs a probability<br>over every token in the vocab</div>
  <div class="arrow">&rarr;</div>
  <div class="step">Decoding strategy<br>picks one token</div>
  <div class="arrow">&rarr;</div>
  <div class="step">Token is appended<br>to the sequence</div>
  <div class="arrow">&rarr;</div>
  <div class="step">Repeat for the<br>next token</div>
</div>

<!-- Notes:
This repeats one token at a time until the model emits an end-of-sequence token or hits a length limit. The choice of decoding strategy is what separates a safe, repetitive answer from a creative — or chaotic — one.
-->

---
<!-- Slide number: 38 -->
# Decoding Strategies Compared

<div class="decode-table-wrap">
  <table class="decode-table">
    <thead>
      <tr><th>Decoding Term</th><th>Core Mechanism / Definition</th><th>Primary Advantage</th><th>Best Use Case</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Greedy Decoding</td>
        <td>Always selects the single token with the highest probability (Max Logit).</td>
        <td>100% deterministic and predictable output.</td>
        <td>Math, code generation, factual QA.</td>
      </tr>
      <tr>
        <td>Temperature</td>
        <td>Scales the probability distribution before Softmax.
          <span class="sub">Low (0.1–0.3): focused &amp; deterministic.</span>
          <span class="sub">High (0.8–1.2): creative &amp; chaotic — risks nonsensical rambling.</span>
        </td>
        <td>Direct control over output creativity &amp; diversity.</td>
        <td>Balancing strict precision with creative variation.</td>
      </tr>
      <tr>
        <td>Top-K Sampling</td>
        <td>Restricts token selection to a fixed count (K) of top candidate tokens (e.g. Top 50).</td>
        <td>Prevents improbable tail tokens &amp; gibberish.</td>
        <td>Controlled open-ended generation.</td>
      </tr>
      <tr>
        <td>Top-P (Nucleus)</td>
        <td>Dynamically keeps adding tokens until their cumulative probability reaches threshold P (e.g. 90%).</td>
        <td>Adaptive candidate pool size based on model confidence.</td>
        <td>Creative writing, chatbots, storytelling.</td>
      </tr>
    </tbody>
  </table>
</div>

<!-- Notes:
Greedy Search always grabs the #1 token — safe and predictable, but repetitive and prone to getting stuck in loops.
Temperature reshapes the whole distribution before sampling — low temperature sharpens it toward the top choice, high temperature flattens it so unlikely tokens get a real chance.
Top-K and Top-P both truncate the candidate pool before sampling, but Top-K uses a fixed cutoff while Top-P adapts dynamically: it stays strict when the model is confident (few tokens needed to reach 90%) and opens up when the model is uncertain (many tokens needed).
-->

---
<!-- Slide number: 39 -->
# Step 2: Detokenization — Rebuilding Text
Once a token ID is picked, the **Detokenizer** translates the numeric ID back into readable characters.

<div class="bpe-frames">
  <div class="frame corpus">
    <div class="frame-label">Mapping</div>
    Matches IDs to sub-word strings.<br>
    ID 15496 &rarr; " un" &nbsp;&middot;&nbsp; ID 995 &rarr; "believ" &nbsp;&middot;&nbsp; ID 1421 &rarr; "able"
  </div>
  <div class="frame counts">
    <div class="frame-label">Merging</div>
    Glues the sub-words together into a single word: " un" + "believ" + "able" &rarr; <strong>"unbelievable"</strong>
  </div>
  <div class="frame step">
    <div class="frame-label">Formatting</div>
    Resolves special symbols, removes byte-level artifacts (like the 'Ġ' whitespace marker), and handles punctuation spacing.
  </div>
</div>

<!-- Notes:
This is the mirror image of BPE encoding from Module 2. Encoding split "unbelievable" into subword pieces; detokenization glues the model's chosen pieces back together.
The 'Ġ' symbol (used by GPT-style byte-level BPE tokenizers) marks "this token starts with a space" — it has to be stripped and converted back into an actual space, not printed as a literal character.
Formatting also cleans up things like spacing before punctuation (e.g. "word ." should render as "word.") so the final text reads naturally.
-->

---
<!-- Slide number: 40 -->
# Summary — The Path to Intelligence

---
<!-- Slide number: 41 -->
<!-- _class: center-content -->
# Discussion (Q & A)

<img src="img/slide32-1.jpeg" alt="Playful illustration of an exploding cloud of ideas" style="max-height:320px;">

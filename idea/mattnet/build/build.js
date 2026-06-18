const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const {
  FaSearchLocation, FaCubes, FaBrain, FaEye, FaLanguage, FaSitemap,
  FaMapMarkerAlt, FaProjectDiagram, FaUser, FaBalanceScale, FaChartBar,
  FaTrophy, FaExclamationTriangle, FaLightbulb, FaLayerGroup, FaCheck,
  FaTimes, FaQuoteLeft, FaPuzzlePiece, FaCrosshairs
} = require("react-icons/fa");

// ---------- palette (Ocean / academic vision-language) ----------
const NAVY = "0B1F3A";      // dominant dark
const DEEP = "12345C";      // deep blue panel
const TEAL = "1C7293";      // secondary
const SEAFOAM = "21A0A0";   // accent
const MINT = "5FD0C5";      // light accent
const ICE = "CFE3EC";       // light text on dark
const CREAM = "F5F8FA";     // light bg
const SLATE = "5A6B78";     // muted text
const INK = "1A2A38";       // body text on light
const GOLD = "F2B705";      // highlight accent

const HFONT = "Microsoft JhengHei"; // 微軟正黑體
const BFONT = "Microsoft JhengHei";

async function iconPng(IconComponent, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
  pres.layout = "W";
  pres.author = "kuo";
  pres.title = "MAttNet 論文導讀";

  const W = 13.333, H = 7.5;

  // pre-render icons
  const ic = {
    search: await iconPng(FaSearchLocation, "#" + MINT),
    cubes: await iconPng(FaCubes, "#" + MINT),
    brain: await iconPng(FaBrain, "#" + MINT),
    eye: await iconPng(FaEye, "#" + SEAFOAM),
    lang: await iconPng(FaLanguage, "#" + SEAFOAM),
    sitemap: await iconPng(FaSitemap, "#" + SEAFOAM),
    user: await iconPng(FaUser, "#FFFFFF"),
    pin: await iconPng(FaMapMarkerAlt, "#FFFFFF"),
    graph: await iconPng(FaProjectDiagram, "#FFFFFF"),
    scale: await iconPng(FaBalanceScale, "#" + SEAFOAM),
    chart: await iconPng(FaChartBar, "#" + SEAFOAM),
    trophy: await iconPng(FaTrophy, "#" + GOLD),
    warn: await iconPng(FaExclamationTriangle, "#" + GOLD),
    bulb: await iconPng(FaLightbulb, "#" + GOLD),
    layers: await iconPng(FaLayerGroup, "#" + SEAFOAM),
    check: await iconPng(FaCheck, "#" + SEAFOAM),
    cross: await iconPng(FaTimes, "#E2706A"),
    quote: await iconPng(FaQuoteLeft, "#" + MINT),
    puzzle: await iconPng(FaPuzzlePiece, "#" + SEAFOAM),
    cross2: await iconPng(FaCrosshairs, "#" + SEAFOAM),
  };

  const shadow = () => ({ type: "outer", color: "0B1F3A", blur: 8, offset: 3, angle: 90, opacity: 0.18 });

  // ===== helper: footer on light slides =====
  function footer(slide, n) {
    slide.addText("MAttNet · CVPR 2018 · 論文導讀", {
      x: 0.5, y: 7.05, w: 8, h: 0.3, fontFace: BFONT, fontSize: 9, color: SLATE, align: "left", margin: 0
    });
    slide.addText(String(n), {
      x: 12.5, y: 7.05, w: 0.5, h: 0.3, fontFace: BFONT, fontSize: 9, color: SLATE, align: "right", margin: 0
    });
  }
  // section eyebrow + title on light slides
  function header(slide, eyebrow, title) {
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 0.55, w: 0.12, h: 0.78, fill: { color: SEAFOAM } });
    slide.addText(eyebrow, { x: 0.78, y: 0.52, w: 11, h: 0.3, fontFace: BFONT, fontSize: 12, color: TEAL, bold: true, charSpacing: 2, margin: 0 });
    slide.addText(title, { x: 0.76, y: 0.78, w: 12, h: 0.62, fontFace: HFONT, fontSize: 30, color: NAVY, bold: true, margin: 0 });
  }

  // =====================================================================
  // SLIDE 1 — TITLE
  // =====================================================================
  let s = pres.addSlide();
  s.background = { color: NAVY };
  // decorative panels
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.18, fill: { color: SEAFOAM } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 7.32, w: W, h: 0.18, fill: { color: TEAL } });
  // faux "bounding box" motif
  s.addShape(pres.shapes.RECTANGLE, { x: 8.7, y: 1.7, w: 3.5, h: 2.4, fill: { type: "none" }, line: { color: MINT, width: 2.5, dashType: "dash" } });
  s.addText("the woman in red\non the left", { x: 8.75, y: 4.15, w: 3.5, h: 0.7, fontFace: BFONT, fontSize: 12, italic: true, color: MINT, align: "center", margin: 0 });
  s.addImage({ data: ic.cross2, x: 10.2, y: 2.75, w: 0.5, h: 0.5 });

  s.addText("CVPR 2018", { x: 0.9, y: 1.5, w: 5, h: 0.4, fontFace: BFONT, fontSize: 14, color: MINT, bold: true, charSpacing: 3, margin: 0 });
  s.addText("MAttNet", { x: 0.82, y: 1.95, w: 8, h: 1.1, fontFace: HFONT, fontSize: 64, color: "FFFFFF", bold: true, margin: 0 });
  s.addText("Modular Attention Network for\nReferring Expression Comprehension", {
    x: 0.9, y: 3.15, w: 7.6, h: 1.0, fontFace: BFONT, fontSize: 20, color: ICE, lineSpacingMultiple: 1.1, margin: 0
  });
  s.addText("模組化注意力網路：把一句指代描述拆成「外觀／位置／關係」三模組，分而治之", {
    x: 0.9, y: 4.35, w: 7.5, h: 0.8, fontFace: BFONT, fontSize: 14, color: MINT, lineSpacingMultiple: 1.2, margin: 0
  });
  s.addText([
    { text: "Licheng Yu · Zhe Lin · Xiaohui Shen · Jimei Yang · Xin Lu · Mohit Bansal · Tamara L. Berg", options: { breakLine: true } },
    { text: "arXiv:1801.08186 · UNC Chapel Hill / Adobe Research", options: {} },
  ], { x: 0.9, y: 5.7, w: 9, h: 0.8, fontFace: BFONT, fontSize: 12, color: ICE, lineSpacingMultiple: 1.3, margin: 0 });

  // =====================================================================
  // SLIDE 2 — 任務：什麼是 Referring Expression Comprehension
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "THE TASK", "任務：指代表達理解 (REC)");
  s.addText("給一張影像 + 一句自然語言描述，模型要在圖中定位出「唯一對應」的那個物件（輸出框，或進一步輸出 mask）。", {
    x: 0.78, y: 1.55, w: 11.7, h: 0.7, fontFace: BFONT, fontSize: 15, color: INK, lineSpacingMultiple: 1.2, margin: 0
  });

  // input/output flow
  const flowY = 2.5;
  // input card
  s.addShape(pres.shapes.RECTANGLE, { x: 0.78, y: flowY, w: 3.5, h: 3.4, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.78, y: flowY, w: 3.5, h: 0.5, fill: { color: DEEP } });
  s.addText("輸入", { x: 0.78, y: flowY, w: 3.5, h: 0.5, fontFace: HFONT, fontSize: 14, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 1.1, y: flowY + 0.75, w: 2.86, h: 1.6, fill: { color: "EAF1F4" }, line: { color: ICE, width: 1 } });
  s.addText("影像 (一群人)", { x: 1.1, y: flowY + 1.4, w: 2.86, h: 0.3, fontFace: BFONT, fontSize: 11, color: SLATE, align: "center", italic: true, margin: 0 });
  s.addText("「the woman in red\non the left」", { x: 0.95, y: flowY + 2.5, w: 3.18, h: 0.7, fontFace: BFONT, fontSize: 15, bold: true, color: TEAL, align: "center", margin: 0 });

  // arrow
  s.addText("➜", { x: 4.4, y: flowY + 1.3, w: 0.9, h: 0.8, fontFace: BFONT, fontSize: 40, color: SEAFOAM, align: "center", valign: "middle", margin: 0 });

  // model card
  s.addShape(pres.shapes.RECTANGLE, { x: 5.35, y: flowY + 0.8, w: 2.6, h: 1.8, fill: { color: NAVY }, shadow: shadow() });
  s.addImage({ data: ic.brain, x: 6.35, y: flowY + 1.1, w: 0.6, h: 0.6 });
  s.addText("MAttNet", { x: 5.35, y: flowY + 1.75, w: 2.6, h: 0.4, fontFace: HFONT, fontSize: 16, bold: true, color: "FFFFFF", align: "center", margin: 0 });
  s.addText("候選框排序", { x: 5.35, y: flowY + 2.12, w: 2.6, h: 0.3, fontFace: BFONT, fontSize: 11, color: MINT, align: "center", margin: 0 });

  s.addText("➜", { x: 8.05, y: flowY + 1.3, w: 0.9, h: 0.8, fontFace: BFONT, fontSize: 40, color: SEAFOAM, align: "center", valign: "middle", margin: 0 });

  // output card
  s.addShape(pres.shapes.RECTANGLE, { x: 9.0, y: flowY, w: 3.5, h: 3.4, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 9.0, y: flowY, w: 3.5, h: 0.5, fill: { color: SEAFOAM } });
  s.addText("輸出", { x: 9.0, y: flowY, w: 3.5, h: 0.5, fontFace: HFONT, fontSize: 14, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 9.32, y: flowY + 0.75, w: 2.86, h: 1.6, fill: { color: "EAF1F4" }, line: { color: ICE, width: 1 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 9.7, y: flowY + 0.95, w: 1.1, h: 1.25, fill: { type: "none" }, line: { color: GOLD, width: 3 } });
  s.addText("命中正確的女人 (IoU ≥ 0.5)", { x: 9.32, y: flowY + 2.5, w: 2.86, h: 0.7, fontFace: BFONT, fontSize: 12, color: INK, align: "center", margin: 0 });

  // contrast strip
  s.addText([
    { text: "與相鄰任務的差別：  ", options: { bold: true, color: NAVY } },
    { text: "物件偵測", options: { bold: true, color: TEAL } }, { text: "=找出所有某類物件 (不分哪一個)  ·  ", options: { color: SLATE } },
    { text: "影像描述", options: { bold: true, color: TEAL } }, { text: "=生成句子 (非定位)  ·  ", options: { color: SLATE } },
    { text: "REC", options: { bold: true, color: SEAFOAM } }, { text: "=給句子→定位唯一物件", options: { color: SLATE } },
  ], { x: 0.78, y: 6.15, w: 11.8, h: 0.5, fontFace: BFONT, fontSize: 12.5, align: "left", margin: 0 });
  footer(s, 2);

  // =====================================================================
  // SLIDE 3 — 核心洞見：為什麼模組化
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "KEY INSIGHT", "核心洞見：為什麼要「模組化」");

  s.addText([
    { text: "過去做法：", options: { bold: true, color: NAVY } },
    { text: "把整句話編碼成 ", options: { color: INK } },
    { text: "一個向量", options: { bold: true, color: "C0392B" } },
    { text: " 去跟影像比對。問題是——一句指代描述其實是 ", options: { color: INK } },
    { text: "異質資訊的混合體", options: { bold: true, color: SEAFOAM } },
    { text: "。", options: { color: INK } },
  ], { x: 0.78, y: 1.55, w: 11.8, h: 0.6, fontFace: BFONT, fontSize: 15, lineSpacingMultiple: 1.2, margin: 0 });

  const cards = [
    { ic: ic.eye, t: "主體外觀", e: "\"red cat\" / \"man with glasses\"", look: "看框「內部」的顏色、類別、屬性", col: TEAL },
    { ic: ic.pin, t: "絕對位置", e: "\"on the left\" / \"top corner\"", look: "看框在整張圖的座標", col: SEAFOAM },
    { ic: ic.graph, t: "物件關係", e: "\"dog next to the table\"", look: "看框「外部」周圍的其他物件", col: DEEP },
  ];
  const cy = 2.45, cw = 3.78, gap = 0.23;
  cards.forEach((c, i) => {
    const cx = 0.78 + i * (cw + gap);
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: cw, h: 2.55, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: cw, h: 0.1, fill: { color: c.col } });
    s.addShape(pres.shapes.OVAL, { x: cx + 0.3, y: cy + 0.32, w: 0.7, h: 0.7, fill: { color: c.col } });
    s.addImage({ data: c.ic, x: cx + 0.45, y: cy + 0.47, w: 0.4, h: 0.4 });
    s.addText(c.t, { x: cx + 1.15, y: cy + 0.4, w: cw - 1.2, h: 0.6, fontFace: HFONT, fontSize: 20, bold: true, color: NAVY, valign: "middle", margin: 0 });
    s.addText(c.e, { x: cx + 0.3, y: cy + 1.25, w: cw - 0.6, h: 0.5, fontFace: BFONT, fontSize: 12.5, italic: true, color: c.col, bold: true, margin: 0 });
    s.addText(c.look, { x: cx + 0.3, y: cy + 1.8, w: cw - 0.6, h: 0.6, fontFace: BFONT, fontSize: 13, color: SLATE, margin: 0 });
  });

  // takeaway band
  s.addShape(pres.shapes.RECTANGLE, { x: 0.78, y: 5.35, w: 11.75, h: 1.25, fill: { color: NAVY }, shadow: shadow() });
  s.addImage({ data: ic.bulb, x: 1.05, y: 5.7, w: 0.55, h: 0.55 });
  s.addText([
    { text: "主張：", options: { bold: true, color: GOLD } },
    { text: "與其用一個大模型硬吃整句，不如拆成三個各司其職的模組，", options: { color: "FFFFFF" } },
    { text: "並讓模型自己學會「哪些字餵給哪個模組、每模組佔多少權重」。", options: { color: MINT, bold: true } },
  ], { x: 1.8, y: 5.55, w: 10.5, h: 0.85, fontFace: BFONT, fontSize: 15, valign: "middle", lineSpacingMultiple: 1.15, margin: 0 });
  footer(s, 3);

  // =====================================================================
  // SLIDE 4 — 整體架構
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 0.55, w: 0.12, h: 0.78, fill: { color: MINT } });
  s.addText("ARCHITECTURE", { x: 0.78, y: 0.52, w: 11, h: 0.3, fontFace: BFONT, fontSize: 12, color: MINT, bold: true, charSpacing: 2, margin: 0 });
  s.addText("整體架構：語言網路 + 三視覺模組", { x: 0.76, y: 0.78, w: 12, h: 0.62, fontFace: HFONT, fontSize: 30, color: "FFFFFF", bold: true, margin: 0 });

  // language box (top)
  s.addShape(pres.shapes.RECTANGLE, { x: 0.9, y: 1.75, w: 5.2, h: 1.95, fill: { color: DEEP }, line: { color: SEAFOAM, width: 1.5 } });
  s.addImage({ data: ic.lang, x: 1.15, y: 1.95, w: 0.5, h: 0.5 });
  s.addText("語言注意力網路", { x: 1.75, y: 1.95, w: 4, h: 0.5, fontFace: HFONT, fontSize: 16, bold: true, color: "FFFFFF", valign: "middle", margin: 0 });
  s.addText([
    { text: "bi-LSTM 編碼每個詞", options: { breakLine: true, bullet: true } },
    { text: "三組詞注意力 → q^subj / q^loc / q^rel", options: { breakLine: true, bullet: true } },
    { text: "模組權重 w_subj / w_loc / w_rel", options: { bullet: true } },
  ], { x: 1.2, y: 2.55, w: 4.7, h: 1.05, fontFace: BFONT, fontSize: 11.5, color: ICE, lineSpacingMultiple: 1.05, margin: 0 });
  s.addText("表達式 r：\"woman in red, left\"", { x: 0.9, y: 3.78, w: 5.2, h: 0.3, fontFace: BFONT, fontSize: 11, italic: true, color: MINT, align: "center", margin: 0 });

  // image/candidate box
  s.addShape(pres.shapes.RECTANGLE, { x: 7.2, y: 1.75, w: 5.2, h: 1.95, fill: { color: DEEP }, line: { color: SEAFOAM, width: 1.5 } });
  s.addImage({ data: ic.eye, x: 7.45, y: 1.95, w: 0.5, h: 0.5 });
  s.addText("視覺骨幹 (Faster R-CNN + ResNet-101)", { x: 8.05, y: 1.95, w: 4.2, h: 0.5, fontFace: HFONT, fontSize: 14, bold: true, color: "FFFFFF", valign: "middle", margin: 0 });
  s.addText([
    { text: "候選物件 o_i 的區域特徵", options: { breakLine: true, bullet: true } },
    { text: "C3 = 低階線索 (顏色、形狀)", options: { breakLine: true, bullet: true } },
    { text: "C4 = 高階線索 (物件類別語意)", options: { bullet: true } },
  ], { x: 7.5, y: 2.55, w: 4.7, h: 1.05, fontFace: BFONT, fontSize: 11.5, color: ICE, lineSpacingMultiple: 1.05, margin: 0 });

  // three modules row
  const mods = [
    { t: "Subject 模組", s: "框內注意力\n外觀 + 屬性", col: TEAL, q: "q^subj" },
    { t: "Location 模組", s: "絕對 + 相對位置\n(5維 + 鄰居位移)", col: SEAFOAM, q: "q^loc" },
    { t: "Relationship 模組", s: "框外注意力\n鄰居 MIL (max)", col: "2E6E8E", q: "q^rel" },
  ];
  const my = 4.4, mw = 3.75, mgap = 0.28;
  mods.forEach((m, i) => {
    const mx = 0.9 + i * (mw + mgap);
    s.addShape(pres.shapes.RECTANGLE, { x: mx, y: my, w: mw, h: 1.5, fill: { color: m.col }, shadow: shadow() });
    s.addText(m.t, { x: mx, y: my + 0.18, w: mw, h: 0.4, fontFace: HFONT, fontSize: 16, bold: true, color: "FFFFFF", align: "center", margin: 0 });
    s.addText(m.s, { x: mx, y: my + 0.62, w: mw, h: 0.7, fontFace: BFONT, fontSize: 12, color: "FFFFFF", align: "center", lineSpacingMultiple: 1.05, margin: 0 });
  });

  // combine bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0.9, y: 6.25, w: 11.5, h: 0.92, fill: { color: GOLD }, shadow: shadow() });
  s.addText("加權合併:   S(o_i | r) = w_subj·S(·|q^subj) + w_loc·S(·|q^loc) + w_rel·S(·|q^rel)", {
    x: 0.9, y: 6.25, w: 11.5, h: 0.92, fontFace: BFONT, fontSize: 16, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0
  });

  // =====================================================================
  // SLIDE 5 — 語言注意力網路
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "LANGUAGE SIDE", "語言注意力網路：學出來的「軟性 parser」");

  // left: steps
  const steps = [
    { n: "1", t: "詞編碼 (bi-LSTM)", d: "每個詞嵌入後過雙向 LSTM，hₜ = [→hₜ , ←hₜ]，融合前後文。" },
    { n: "2", t: "詞層級注意力 ×3", d: "三個可訓練向量 f_m 對每個詞算 softmax，加權出 q^subj / q^loc / q^rel 三段 phrase embedding。" },
    { n: "3", t: "模組權重", d: "首尾 hidden 串接過 FC+softmax → [w_subj, w_loc, w_rel]，決定各模組貢獻度。" },
  ];
  let sy = 1.6;
  steps.forEach((st) => {
    s.addShape(pres.shapes.OVAL, { x: 0.78, y: sy, w: 0.6, h: 0.6, fill: { color: SEAFOAM } });
    s.addText(st.n, { x: 0.78, y: sy, w: 0.6, h: 0.6, fontFace: HFONT, fontSize: 22, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
    s.addText(st.t, { x: 1.55, y: sy - 0.05, w: 6.0, h: 0.4, fontFace: HFONT, fontSize: 17, bold: true, color: NAVY, margin: 0 });
    s.addText(st.d, { x: 1.55, y: sy + 0.38, w: 6.1, h: 0.85, fontFace: BFONT, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.12, valign: "top", margin: 0 });
    sy += 1.32;
  });

  // right: example attention viz
  s.addShape(pres.shapes.RECTANGLE, { x: 8.0, y: 1.65, w: 4.55, h: 3.6, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
  s.addText("詞注意力示意", { x: 8.0, y: 1.8, w: 4.55, h: 0.4, fontFace: HFONT, fontSize: 14, bold: true, color: TEAL, align: "center", margin: 0 });
  const words = [
    { w: "woman", subj: 0.9, loc: 0.1 },
    { w: "in", subj: 0.2, loc: 0.1 },
    { w: "red", subj: 0.95, loc: 0.05 },
    { w: "on", subj: 0.1, loc: 0.4 },
    { w: "left", subj: 0.05, loc: 0.95 },
  ];
  let wy = 2.4;
  words.forEach((wd) => {
    s.addText(wd.w, { x: 8.2, y: wy, w: 1.2, h: 0.35, fontFace: BFONT, fontSize: 13, bold: true, color: INK, valign: "middle", margin: 0 });
    // subj bar
    s.addShape(pres.shapes.RECTANGLE, { x: 9.45, y: wy + 0.04, w: 1.4 * wd.subj, h: 0.12, fill: { color: TEAL } });
    // loc bar
    s.addShape(pres.shapes.RECTANGLE, { x: 9.45, y: wy + 0.2, w: 1.4 * wd.loc, h: 0.12, fill: { color: SEAFOAM } });
    wy += 0.5;
  });
  s.addText([
    { text: "■ ", options: { color: TEAL, bold: true } }, { text: "subject  ", options: { color: SLATE } },
    { text: "■ ", options: { color: SEAFOAM, bold: true } }, { text: "location", options: { color: SLATE } },
  ], { x: 8.2, y: 4.95, w: 4.2, h: 0.3, fontFace: BFONT, fontSize: 11, align: "center", margin: 0 });

  // bottom highlight
  s.addShape(pres.shapes.RECTANGLE, { x: 0.78, y: 5.55, w: 11.75, h: 1.05, fill: { color: NAVY }, shadow: shadow() });
  s.addImage({ data: ic.trophy, x: 1.05, y: 5.82, w: 0.5, h: 0.5 });
  s.addText([
    { text: "為何重要：", options: { bold: true, color: GOLD } },
    { text: "這個拆分完全是端到端「學」出來的，不靠外部語法分析器。實驗顯示比接外部模板 parser ", options: { color: "FFFFFF" } },
    { text: "高約 5%", options: { bold: true, color: MINT } },
    { text: "(parser 會犯錯且錯誤會傳遞)。", options: { color: "FFFFFF" } },
  ], { x: 1.75, y: 5.62, w: 10.6, h: 0.9, fontFace: BFONT, fontSize: 14, valign: "middle", lineSpacingMultiple: 1.1, margin: 0 });
  footer(s, 5);

  // =====================================================================
  // SLIDE 6 — 三個視覺模組 (detail)
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "VISION SIDE", "三個視覺模組：各司其職");

  const vmods = [
    {
      t: "Subject 模組", tag: "框內注意力 (in-box)", col: TEAL, ic: ic.eye,
      pts: ["屬性預測：C3+C4 → 1×1 conv，加權 BCE 平衡長尾屬性", "Phrase-guided 注意力池化：用 q^subj 在 14×14 網格挑相關部位 (如「衣服」)", "解決「是什麼、什麼顏色」"]
    },
    {
      t: "Location 模組", tag: "絕對 + 相對位置", col: SEAFOAM, ic: ic.pin,
      pts: ["絕對位置：5 維向量 (左上、右下座標 + 面積比)", "相對位置 (dif)：周圍 5 個同類物件的位移與面積比", "★ 消融中單項貢獻最大 (+2.4pp)"]
    },
    {
      t: "Relationship 模組", tag: "框外注意力 (out-of-box)", col: DEEP, ic: ic.graph,
      pts: ["看周圍 5 個任意類別鄰居 (外觀 + 位移)", "弱監督 MIL：不知指哪個鄰居 → 對所有鄰居取 max", "解決「桌子旁邊的狗」這類關係"]
    },
  ];
  const vy = 1.65, vw = 3.78, vgap = 0.23;
  vmods.forEach((m, i) => {
    const vx = 0.78 + i * (vw + vgap);
    s.addShape(pres.shapes.RECTANGLE, { x: vx, y: vy, w: vw, h: 4.9, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: vx, y: vy, w: vw, h: 1.15, fill: { color: m.col } });
    s.addImage({ data: m.ic, x: vx + 0.32, y: vy + 0.3, w: 0.55, h: 0.55 });
    s.addText(m.t, { x: vx + 1.0, y: vy + 0.22, w: vw - 1.1, h: 0.45, fontFace: HFONT, fontSize: 17, bold: true, color: "FFFFFF", margin: 0 });
    s.addText(m.tag, { x: vx + 1.0, y: vy + 0.68, w: vw - 1.1, h: 0.35, fontFace: BFONT, fontSize: 11.5, color: ICE, italic: true, margin: 0 });
    s.addText(
      m.pts.map((p, j) => ({ text: p, options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 10 } })),
      { x: vx + 0.32, y: vy + 1.45, w: vw - 0.6, h: 3.3, fontFace: BFONT, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.1, valign: "top", margin: 0 }
    );
  });
  footer(s, 6);

  // =====================================================================
  // SLIDE 7 — 匹配函數與損失
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "SCORING & TRAINING", "匹配函數與損失：如何打分、如何訓練");

  // left: matching function
  s.addShape(pres.shapes.RECTANGLE, { x: 0.78, y: 1.7, w: 5.7, h: 2.3, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
  s.addImage({ data: ic.scale, x: 1.05, y: 1.95, w: 0.5, h: 0.5 });
  s.addText("匹配函數 F (三模組共用)", { x: 1.65, y: 1.97, w: 4.6, h: 0.45, fontFace: HFONT, fontSize: 16, bold: true, color: NAVY, valign: "middle", margin: 0 });
  s.addText([
    { text: "視覺特徵與語言 phrase 各過一個 MLP，投影到", options: { breakLine: true } },
    { text: "共同語意空間", options: { bold: true, color: SEAFOAM, breakLine: true } },
    { text: "→ L2 normalize → 取內積當匹配分數。", options: {} },
  ], { x: 1.05, y: 2.55, w: 5.2, h: 1.0, fontFace: BFONT, fontSize: 13, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 1.05, y: 3.45, w: 5.15, h: 0.42, fill: { color: "EAF1F4" } });
  s.addText("F = ⟨ L2(MLPᵥ(ṽ)) , L2(MLP_q(q)) ⟩", { x: 1.05, y: 3.45, w: 5.15, h: 0.42, fontFace: "Consolas", fontSize: 12.5, color: TEAL, bold: true, align: "center", valign: "middle", margin: 0 });

  // right: ranking loss
  s.addShape(pres.shapes.RECTANGLE, { x: 6.83, y: 1.7, w: 5.7, h: 2.3, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
  s.addImage({ data: ic.balance || ic.scale, x: 7.1, y: 1.95, w: 0.5, h: 0.5 });
  s.addText("Ranking Loss (兩個 hard negative)", { x: 7.7, y: 1.97, w: 4.6, h: 0.45, fontFace: HFONT, fontSize: 16, bold: true, color: NAVY, valign: "middle", margin: 0 });
  s.addText([
    { text: "每個正配對 (o_i, r_i) 配兩個負例：", options: { breakLine: true } },
    { text: "• 同物件配錯句子 r_j     • 同句子配錯物件 o_k", options: { breakLine: true, color: TEAL, bold: true } },
    { text: "要求正配對分數高出負例至少一個 margin Δ。", options: {} },
  ], { x: 7.1, y: 2.55, w: 5.2, h: 1.3, fontFace: BFONT, fontSize: 13, color: INK, lineSpacingMultiple: 1.2, margin: 0 });

  // total loss band
  s.addShape(pres.shapes.RECTANGLE, { x: 0.78, y: 4.25, w: 11.75, h: 0.75, fill: { color: DEEP } });
  s.addText("總損失：  L = L_subj^attr  (屬性預測)  +  L_rank  (排序)", { x: 0.78, y: 4.25, w: 11.75, h: 0.75, fontFace: BFONT, fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });

  // emphasis box
  s.addShape(pres.shapes.RECTANGLE, { x: 0.78, y: 5.3, w: 11.75, h: 1.3, fill: { color: NAVY }, shadow: shadow() });
  s.addImage({ data: ic.bulb, x: 1.05, y: 5.68, w: 0.55, h: 0.55 });
  s.addText([
    { text: "弱監督的威力：", options: { bold: true, color: GOLD } },
    { text: "整套系統——詞注意力、模組權重、框內空間注意力、框外鄰居 MIL——", options: { color: "FFFFFF" } },
    { text: "全部僅靠「物件–表達式配對」端到端訓練。", options: { bold: true, color: MINT } },
    { text: "沒有人標哪個字屬於哪模組、注意力該看哪——這些都自動湧現並帶來可解釋性。", options: { color: "FFFFFF" } },
  ], { x: 1.8, y: 5.4, w: 10.5, h: 1.1, fontFace: BFONT, fontSize: 14, valign: "middle", lineSpacingMultiple: 1.15, margin: 0 });
  footer(s, 7);

  // =====================================================================
  // SLIDE 8 — 資料集
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "DATASETS", "三個評測資料集 (皆建於 MS COCO)");

  const dsRows = [
    ["資料集", "收集方式", "平均句長", "同類物件", "特點"],
    ["RefCOCO", "互動遊戲", "~3.5 詞", "3.9", "句子短，常用位置詞"],
    ["RefCOCO+", "互動遊戲", "~3.5 詞", "3.9", "禁絕對位置詞 → 逼模型靠外觀"],
    ["RefCOCOg", "非互動", "~8.4 詞", "1.63", "句子長，常描述關係"],
  ];
  const tRows = dsRows.map((row, ri) => row.map((c) => ({
    text: c,
    options: ri === 0
      ? { fill: { color: NAVY }, color: "FFFFFF", bold: true, fontSize: 14, align: "center", valign: "middle", fontFace: HFONT }
      : { fill: { color: ri % 2 ? "FFFFFF" : "EAF1F4" }, color: INK, fontSize: 13, align: c === row[0] ? "left" : "center", valign: "middle", bold: ri > 0 && c === row[0], fontFace: BFONT }
  })));
  s.addTable(tRows, {
    x: 0.78, y: 1.7, w: 11.75, colW: [2.2, 2.3, 2.0, 1.8, 3.45], rowH: [0.55, 0.7, 0.7, 0.7],
    border: { pt: 1, color: ICE }, valign: "middle",
  });

  // testA/testB explanation
  s.addShape(pres.shapes.RECTANGLE, { x: 0.78, y: 4.7, w: 5.75, h: 1.85, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.78, y: 4.7, w: 5.75, h: 0.55, fill: { color: TEAL } });
  s.addText("testA", { x: 0.95, y: 4.7, w: 5.4, h: 0.55, fontFace: HFONT, fontSize: 16, bold: true, color: "FFFFFF", valign: "middle", margin: 0 });
  s.addImage({ data: ic.user, x: 5.7, y: 4.78, w: 0.4, h: 0.4 });
  s.addText("含多個「人」的影像\n→ 考驗區分相似的人 (外觀細節)", { x: 0.95, y: 5.4, w: 5.4, h: 1.0, fontFace: BFONT, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, { x: 6.78, y: 4.7, w: 5.75, h: 1.85, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.78, y: 4.7, w: 5.75, h: 0.55, fill: { color: SEAFOAM } });
  s.addText("testB", { x: 6.95, y: 4.7, w: 5.4, h: 0.55, fontFace: HFONT, fontSize: 16, bold: true, color: "FFFFFF", valign: "middle", margin: 0 });
  s.addImage({ data: ic.cubes, x: 11.7, y: 4.78, w: 0.4, h: 0.4 });
  s.addText("含多個「其他類別」物件的影像\n→ 考驗區分各式物件", { x: 6.95, y: 5.4, w: 5.4, h: 1.0, fontFace: BFONT, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
  footer(s, 8);

  // =====================================================================
  // SLIDE 9 — 主結果 (chart)
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "MAIN RESULTS", "主結果：大幅領先當時 SOTA");
  s.addText("使用 GT 候選框，準確率 (%)。對照當時最強前作 Speaker+Listener+Reinforcer (RefCOCO val ≈ 79.6)，MAttNet 領先約 10pp。", {
    x: 0.78, y: 1.5, w: 11.7, h: 0.55, fontFace: BFONT, fontSize: 13, color: INK, lineSpacingMultiple: 1.15, margin: 0
  });

  s.addChart(pres.charts.BAR, [
    { name: "val", labels: ["RefCOCO", "RefCOCO+", "RefCOCOg"], values: [85.65, 71.01, 78.10] },
    { name: "testA / test", labels: ["RefCOCO", "RefCOCO+", "RefCOCOg"], values: [85.26, 75.13, 78.12] },
    { name: "testB", labels: ["RefCOCO", "RefCOCO+", "RefCOCOg"], values: [84.57, 66.17, null] },
  ], {
    x: 0.78, y: 2.2, w: 7.6, h: 4.4, barDir: "col",
    chartColors: [TEAL, SEAFOAM, MINT],
    chartArea: { fill: { color: "FFFFFF" } },
    catAxisLabelColor: INK, catAxisLabelFontSize: 13, catAxisLabelFontFace: BFONT,
    valAxisLabelColor: SLATE, valAxisMinVal: 50, valAxisMaxVal: 90, valAxisMajorUnit: 10,
    valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 9, dataLabelFontBold: true,
    showLegend: true, legendPos: "t", legendColor: INK, legendFontSize: 12,
    showTitle: false,
  });

  // side stat callouts
  const stats = [
    { n: "85.7", l: "RefCOCO val\n準確率 (%)", col: TEAL },
    { n: "~10pp", l: "領先當時\nSOTA", col: SEAFOAM },
    { n: "3", l: "資料集全部\n刷新紀錄", col: DEEP },
  ];
  let sty = 2.35;
  stats.forEach((st) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 8.75, y: sty, w: 3.78, h: 1.32, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: 8.75, y: sty, w: 0.12, h: 1.32, fill: { color: st.col } });
    s.addText(st.n, { x: 9.05, y: sty + 0.12, w: 1.9, h: 1.1, fontFace: HFONT, fontSize: 36, bold: true, color: st.col, valign: "middle", margin: 0 });
    s.addText(st.l, { x: 10.85, y: sty + 0.12, w: 1.6, h: 1.1, fontFace: BFONT, fontSize: 12, color: SLATE, valign: "middle", lineSpacingMultiple: 1.1, margin: 0 });
    sty += 1.45;
  });
  footer(s, 9);

  // =====================================================================
  // SLIDE 10 — 消融 (chart)
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "ABLATION", "消融分析：每個設計值多少分");
  s.addText("RefCOCO val 上逐步堆疊模組 (res101-frcn)。相對位置 (loc dif) 與框內注意力 (attn) 貢獻最大。", {
    x: 0.78, y: 1.5, w: 11.7, h: 0.5, fontFace: BFONT, fontSize: 13, color: INK, margin: 0
  });

  s.addChart(pres.charts.BAR, [
    { name: "RefCOCO val (%)", labels: ["baseline\n(subj+loc)", "MAttNet\n(subj+loc)", "+loc(dif)", "+rel", "+attr", "+attn\n(完整)"], values: [79.14, 79.68, 82.06, 82.54, 83.54, 85.65] },
  ], {
    x: 0.78, y: 2.15, w: 8.0, h: 4.5, barDir: "col",
    chartColors: [SEAFOAM],
    chartArea: { fill: { color: "FFFFFF" } },
    catAxisLabelColor: INK, catAxisLabelFontSize: 11, catAxisLabelFontFace: BFONT,
    valAxisLabelColor: SLATE, valAxisMinVal: 76, valAxisMaxVal: 87, valAxisMajorUnit: 2,
    valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: NAVY, dataLabelFontSize: 11, dataLabelFontBold: true,
    showLegend: false, showTitle: false,
  });

  // findings
  s.addShape(pres.shapes.RECTANGLE, { x: 9.05, y: 2.15, w: 3.48, h: 4.5, fill: { color: NAVY }, shadow: shadow() });
  s.addText("關鍵發現", { x: 9.3, y: 2.35, w: 3.0, h: 0.45, fontFace: HFONT, fontSize: 17, bold: true, color: GOLD, margin: 0 });
  s.addText([
    { text: "模組化本身就贏 baseline", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 12, bold: true, color: "FFFFFF" } },
    { text: "loc(dif) 相對位置 +2.4pp，單項最大", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 12, color: ICE } },
    { text: "attn 框內注意力 +2.1pp，對「人」類別尤其有效", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 12, color: ICE } },
    { text: "學出來的軟 parser 勝外部模板 parser 約 5%", options: { bullet: { code: "2022" }, color: MINT, bold: true } },
  ], { x: 9.3, y: 2.95, w: 3.0, h: 3.6, fontFace: BFONT, fontSize: 12.5, lineSpacingMultiple: 1.1, valign: "top", margin: 0 });
  footer(s, 10);

  // =====================================================================
  // SLIDE 11 — 分割延伸 + 全自動偵測
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "EXTENSIONS", "延伸：像素分割 + 全自動偵測");

  // left: segmentation chart
  s.addText("換 Mask R-CNN：把選中的框餵給 mask 分支 (理解與分割「解耦」)。像素精度幾乎翻倍。", {
    x: 0.78, y: 1.5, w: 7.3, h: 0.6, fontFace: BFONT, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.15, margin: 0
  });
  s.addChart(pres.charts.BAR, [
    { name: "MAttNet", labels: ["RefCOCO val", "RefCOCO testA", "RefCOCO+ val", "RefCOCO+ testA"], values: [56.51, 62.37, 46.67, 52.39] },
    { name: "前作 (D+RMI+DCRF)", labels: ["RefCOCO val", "RefCOCO testA", "RefCOCO+ val", "RefCOCO+ testA"], values: [45.18, 45.69, 29.86, 30.48] },
  ], {
    x: 0.6, y: 2.2, w: 7.5, h: 4.4, barDir: "col",
    chartColors: [SEAFOAM, "B9C7D0"],
    chartArea: { fill: { color: "FFFFFF" } },
    catAxisLabelColor: INK, catAxisLabelFontSize: 10, catAxisLabelFontFace: BFONT,
    valAxisLabelColor: SLATE, valAxisMinVal: 0, valAxisMaxVal: 70, valAxisMajorUnit: 20, valAxisTitle: "IoU",
    valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: NAVY, dataLabelFontSize: 9, dataLabelFontBold: true,
    showLegend: true, legendPos: "t", legendColor: INK, legendFontSize: 11, showTitle: false,
  });

  // right: auto-detection panel
  s.addShape(pres.shapes.RECTANGLE, { x: 8.45, y: 2.2, w: 4.08, h: 4.4, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 8.45, y: 2.2, w: 4.08, h: 0.6, fill: { color: DEEP } });
  s.addText("全自動偵測 (用偵測框非 GT)", { x: 8.6, y: 2.2, w: 3.8, h: 0.6, fontFace: HFONT, fontSize: 13.5, bold: true, color: "FFFFFF", valign: "middle", margin: 0 });
  s.addText("RefCOCO (res101-mrcn)", { x: 8.7, y: 2.95, w: 3.6, h: 0.35, fontFace: BFONT, fontSize: 12, color: SLATE, margin: 0 });
  const det = [["val", "76.65"], ["testA", "81.14"], ["testB", "69.99"]];
  let dy = 3.4;
  det.forEach(([k, v]) => {
    s.addText(k, { x: 8.7, y: dy, w: 1.5, h: 0.5, fontFace: BFONT, fontSize: 14, color: INK, valign: "middle", margin: 0 });
    s.addText(v, { x: 10.2, y: dy, w: 2.1, h: 0.5, fontFace: HFONT, fontSize: 22, bold: true, color: SEAFOAM, align: "right", valign: "middle", margin: 0 });
    dy += 0.6;
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 8.7, y: 5.3, w: 3.6, h: 1.1, fill: { color: "EAF1F4" } });
  s.addText("模組改善趨勢與用 GT 框時一致；Mask R-CNN 偵測品質優於 Faster R-CNN。", {
    x: 8.85, y: 5.4, w: 3.3, h: 0.95, fontFace: BFONT, fontSize: 12, color: INK, lineSpacingMultiple: 1.15, valign: "middle", margin: 0
  });
  footer(s, 11);

  // =====================================================================
  // SLIDE 12 — 貢獻與侷限
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: CREAM };
  header(s, "CONTRIBUTIONS & LIMITS", "貢獻與侷限");

  // contributions
  s.addShape(pres.shapes.RECTANGLE, { x: 0.78, y: 1.7, w: 5.75, h: 4.85, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.78, y: 1.7, w: 5.75, h: 0.62, fill: { color: SEAFOAM } });
  s.addImage({ data: ic.check, x: 0.98, y: 1.78, w: 0.45, h: 0.45 });
  s.addText("主要貢獻", { x: 1.55, y: 1.7, w: 4.8, h: 0.62, fontFace: HFONT, fontSize: 18, bold: true, color: "FFFFFF", valign: "middle", margin: 0 });
  s.addText([
    { text: "模組化分解：明確拆成 subject / location / relationship 三模組", options: { bullet: { code: "2713" }, breakLine: true, paraSpaceAfter: 14 } },
    { text: "雙重注意力：語言端 (詞+模組權重) + 視覺端 (框內+框外)", options: { bullet: { code: "2713" }, breakLine: true, paraSpaceAfter: 14 } },
    { text: "端到端弱監督：注意力與分工自動湧現，附帶可解釋性", options: { bullet: { code: "2713" }, breakLine: true, paraSpaceAfter: 14 } },
    { text: "SOTA + 通用：bbox 與 pixel 兩層級、三資料集全面刷新", options: { bullet: { code: "2713" } } },
  ], { x: 1.05, y: 2.55, w: 5.25, h: 3.8, fontFace: BFONT, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.15, valign: "top", margin: 0 });

  // limitations
  s.addShape(pres.shapes.RECTANGLE, { x: 6.78, y: 1.7, w: 5.75, h: 4.85, fill: { color: "FFFFFF" }, line: { color: ICE, width: 1 }, shadow: shadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.78, y: 1.7, w: 5.75, h: 0.62, fill: { color: "C0625A" } });
  s.addImage({ data: ic.warn, x: 6.98, y: 1.78, w: 0.45, h: 0.45 });
  s.addText("侷限 (也是後續研究的縫)", { x: 7.55, y: 1.7, w: 4.8, h: 0.62, fontFace: HFONT, fontSize: 18, bold: true, color: "FFFFFF", valign: "middle", margin: 0 });
  s.addText([
    { text: "預設目標存在且唯一：強制輸出最高分框，無「棄答 / 多目標」概念", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 13, bold: true, color: "A23E37" } },
    { text: "依賴候選框品質：two-stage 排序，效能受偵測器召回限制", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 13, color: INK } },
    { text: "固定三類模組：對巢狀 / 多跳關係的表達力有限", options: { bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 13, color: INK } },
    { text: "誤差來源：訓練資料稀疏、表達式歧義、偵測失敗", options: { bullet: { code: "2022" }, color: INK } },
  ], { x: 7.05, y: 2.55, w: 5.25, h: 3.8, fontFace: BFONT, fontSize: 13.5, lineSpacingMultiple: 1.15, valign: "top", margin: 0 });
  footer(s, 12);

  // =====================================================================
  // SLIDE 13 — 一句話總結 (dark closing)
  // =====================================================================
  s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.18, fill: { color: SEAFOAM } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 7.32, w: W, h: 0.18, fill: { color: TEAL } });
  s.addImage({ data: ic.quote, x: 0.95, y: 1.5, w: 0.9, h: 0.9 });
  s.addText("TAKEAWAY", { x: 2.05, y: 1.55, w: 8, h: 0.4, fontFace: BFONT, fontSize: 14, color: MINT, bold: true, charSpacing: 3, margin: 0 });
  s.addText("一句話總結", { x: 2.0, y: 1.9, w: 9, h: 0.7, fontFace: HFONT, fontSize: 34, bold: true, color: "FFFFFF", margin: 0 });

  s.addText([
    { text: "MAttNet 把一句指代描述「軟性地」拆成三塊——", options: { color: "FFFFFF" } },
    { text: "主體外觀 / 絕對位置 / 與鄰居的關係", options: { color: MINT, bold: true } },
    { text: "——分別交給三個視覺模組各算匹配分數，再用學出來的權重動態合併。", options: { color: "FFFFFF" } },
  ], { x: 1.0, y: 3.0, w: 11.3, h: 1.5, fontFace: BFONT, fontSize: 22, lineSpacingMultiple: 1.3, margin: 0 });

  s.addText([
    { text: "「分而治之」比把整句揉成單一向量好很多 (RefCOCO 系列領先約 10pp)，", options: { color: ICE } },
    { text: "且天生具備可解釋性：看得出每個字、每塊影像區域對哪個模組起了作用。", options: { color: ICE } },
  ], { x: 1.0, y: 4.7, w: 11.3, h: 1.1, fontFace: BFONT, fontSize: 16, lineSpacingMultiple: 1.3, margin: 0 });

  // three keyword chips
  const chips = ["模組化分解", "雙重注意力", "端到端弱監督"];
  chips.forEach((c, i) => {
    const cx = 1.0 + i * 3.6;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: 6.0, w: 3.3, h: 0.7, fill: { color: DEEP }, line: { color: SEAFOAM, width: 1.5 }, rectRadius: 0.35 });
    s.addText(c, { x: cx, y: 6.0, w: 3.3, h: 0.7, fontFace: HFONT, fontSize: 16, bold: true, color: MINT, align: "center", valign: "middle", margin: 0 });
  });

  await pres.writeFile({ fileName: "C:/Users/kuo/Desktop/paperwork/idea/mattnet/MAttNet_分享簡報.pptx" });
  console.log("DONE");
})();

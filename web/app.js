const TEXT = {
  en: {
    title: "Crack Gauge",
    sub: "Trace a crack in a photo and measure how wide it is",
    drop: "Drop a crack photo here",
    hint: "or click to choose. Put a 10-baht coin next to the crack to get millimetres.",
    examples: "Try an example:",
    coinNote: "Examples are held-out test photos. The coin in the last one is drawn in to show the scale.",
    verdict: "Result",
    width: "typical width (90th pct.)",
    max: "widest point",
    length: "length traced",
    time: "seconds",
    scaleTitle: "Crack width guide (BRE Digest 251)",
    howTitle: "How it works",
    how: "A U-Net model traces the crack pixel by pixel. The edges are tightened to the dark gap, then the width is measured along the crack's centre line. The coin sets the scale: a 10-baht coin is 26 mm across.",
    warn: "Estimates only. This does not replace a structural engineer.",
    credit: "Training photos: Concrete Crack Segmentation Dataset (Özgenel, Mendeley Data, CC BY 4.0)",
    noCrack: "No crack found", noCrackSub: "Try a closer, sharper photo.",
    noCoin: "Crack traced", noCoinSub: "No coin found, so widths are in pixels. Add a 10-baht coin for millimetres.",
    cats: [["0", "Hairline", "< 0.1 mm"], ["1", "Very slight", "< 1 mm"], ["2", "Slight", "1–5 mm"],
           ["3", "Moderate · ask an engineer", "5–15 mm"], ["4", "Severe", "15–25 mm"], ["5", "Very severe", "> 25 mm"]],
    px: "px", mm: "mm",
  },
  th: {
    title: "Crack Gauge",
    sub: "หารอยร้าวในรูป แล้ววัดว่ากว้างกี่มิลลิเมตร",
    drop: "ลากรูปรอยร้าวมาวางตรงนี้",
    hint: "หรือกดเลือกรูป · วางเหรียญ 10 บาทไว้ข้างรอยร้าว จะได้ค่าเป็นมิลลิเมตร",
    examples: "ลองรูปตัวอย่าง:",
    coinNote: "รูปตัวอย่างมาจากชุดทดสอบที่โมเดลไม่เคยเห็น · เหรียญในรูปสุดท้ายเป็นภาพวาดเพื่อสาธิตการวัด",
    verdict: "ผล",
    width: "ความกว้างทั่วไป (เปอร์เซ็นไทล์ 90)",
    max: "จุดที่กว้างสุด",
    length: "ความยาวที่ตามได้",
    time: "วินาที",
    scaleTitle: "ระดับรอยร้าวตามความกว้าง (BRE Digest 251)",
    howTitle: "ทำงานยังไง",
    how: "โมเดล U-Net ระบายรอยร้าวทีละพิกเซล แล้วปรับขอบให้เหลือเฉพาะร่องที่มืดจริง จากนั้นวัดความกว้างตามแนวกลางรอย เหรียญใช้เป็นไม้บรรทัด เหรียญ 10 บาทกว้าง 26 มม.",
    warn: "เป็นค่าประมาณ ไม่ใช้แทนวิศวกรโครงสร้าง",
    credit: "ภาพที่ใช้ฝึก: Concrete Crack Segmentation Dataset (Özgenel, Mendeley Data, CC BY 4.0)",
    noCrack: "ไม่พบรอยร้าว", noCrackSub: "ลองถ่ายใกล้ขึ้นและให้ภาพคมกว่านี้",
    noCoin: "เจอรอยร้าว", noCoinSub: "ไม่เจอเหรียญ ค่าจึงเป็นพิกเซล · วางเหรียญ 10 บาทเพื่อได้ค่าเป็นมิลลิเมตร",
    cats: [["0", "ผิวปูน", "< 0.1 มม."], ["1", "เล็กมาก", "< 1 มม."], ["2", "เล็ก", "1–5 มม."],
           ["3", "ปานกลาง · ควรให้วิศวกรดู", "5–15 มม."], ["4", "รุนแรง", "15–25 มม."], ["5", "รุนแรงมาก", "> 25 มม."]],
    px: "พิกเซล", mm: "มม.",
  },
};
const EXAMPLES = ["examples/1.jpg", "examples/2.jpg", "examples/3.jpg", "examples/coin.jpg"];

let lang = "en";
try { lang = localStorage.getItem("crackgauge-lang") || "en"; } catch (e) {}
let last = null;
const $ = id => document.getElementById(id);
const t = k => TEXT[lang][k];

function applyText() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-t]").forEach(el => (el.textContent = t(el.dataset.t)));
  $("lang").textContent = lang === "en" ? "ไทย" : "EN";
  renderCats(last && last.category);
  if (last) render(last);
}

function renderCats(active) {
  $("cats").innerHTML = t("cats").map(([c, name, range]) =>
    `<li class="${c === active ? "on" : ""}"><b>${c}</b><span>${name}</span><span>${range}</span></li>`).join("");
}

function render(r) {
  last = r;
  if (r.image) { $("preview").src = r.image; $("preview").hidden = false; }
  const v = $("verdict");
  v.className = "card verdict";
  $("sTime").textContent = r.seconds ?? "–";
  if (!r.found) {
    $("vMain").textContent = t("noCrack"); $("vSub").textContent = t("noCrackSub");
    ["sWidth", "sMax", "sLen"].forEach(id => ($(id).textContent = "–"));
    renderCats(null);
    return;
  }
  if (r.coin && r.p90_mm != null) {
    const cat = t("cats").find(c => c[0] === r.category);
    $("vMain").textContent = `${r.p90_mm} ${t("mm")} · ${cat[1]}`;
    $("vSub").textContent = cat[2];
    v.classList.add(+r.category >= 3 ? "bad" : +r.category >= 2 ? "warn" : "ok");
    $("sWidth").textContent = `${r.p90_mm} ${t("mm")}`;
    $("sMax").textContent = `${r.max_mm} ${t("mm")}`;
    $("sLen").textContent = `${r.length_mm} ${t("mm")}`;
    renderCats(r.category);
  } else {
    $("vMain").textContent = t("noCoin"); $("vSub").textContent = t("noCoinSub");
    $("sWidth").textContent = `${r.p90_px} ${t("px")}`;
    $("sMax").textContent = `${r.max_px} ${t("px")}`;
    $("sLen").textContent = `${r.length_px} ${t("px")}`;
    renderCats(null);
  }
}

async function send(blob) {
  $("busy").hidden = false;
  $("dropInner").hidden = true;
  $("preview").src = URL.createObjectURL(blob);
  $("preview").hidden = false;
  const fd = new FormData();
  fd.append("file", blob, "photo.jpg");
  try {
    const res = await fetch("/api/measure", { method: "POST", body: fd });
    render(await res.json());
  } finally {
    $("busy").hidden = true;
  }
}

$("file").onchange = e => e.target.files[0] && send(e.target.files[0]);
const drop = $("drop");
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); e.dataTransfer.files[0] && send(e.dataTransfer.files[0]); };
$("lang").onclick = () => {
  lang = lang === "en" ? "th" : "en";
  try { localStorage.setItem("crackgauge-lang", lang); } catch (e) {}
  applyText();
};

$("exampleList").innerHTML = EXAMPLES.map(src => `<img class="ex" src="${src}" alt="" data-src="${src}">`).join("");
document.querySelectorAll(".ex").forEach(el => (el.onclick = async e => {
  e.preventDefault();
  send(await (await fetch(el.dataset.src)).blob());
}));

applyText();
const auto = new URLSearchParams(location.search).get("example");
if (auto) fetch(EXAMPLES[+auto]).then(r => r.blob()).then(send);

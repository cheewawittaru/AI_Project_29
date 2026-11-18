import io, json, os, time
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader, PdfWriter

st.set_page_config(page_title="AI Budget Advisor", page_icon="💰", layout="wide")

# ---------------- ตั้งค่า ----------------
CATS = ["อาหารและเครื่องดื่ม", "ของใช้และซูเปอร์มาร์เก็ต", "เดินทาง", "ช้อปปิ้ง",
        "การเรียน", "ที่พักและสาธารณูปโภค", "บันเทิงและสมาชิก", "สุขภาพ",
        "โอนให้บุคคล/อื่นๆ", "รายรับ"]
OTHER = "โอนให้บุคคล/อื่นๆ"
INCOME = "รายรับ"
EXPENSE_CATS = [c for c in CATS if c != INCOME]
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]  # ปรับตามรายชื่อโมเดลที่ใช้ได้
FEEDBACK_URL = ""  # ใส่ลิงก์ Google Form แบบประเมินที่นี่

# กฎร้านค้า/คำที่รู้จัก (ตรวจก่อน AI) แก้เพิ่มได้ตามนิยามของกลุ่ม
RULES = {
    "7-eleven": "ของใช้และซูเปอร์มาร์เก็ต", "เซเว่น": "ของใช้และซูเปอร์มาร์เก็ต",
    "ค่าไฟ": "ที่พักและสาธารณูปโภค", "ค่าน้ำ": "ที่พักและสาธารณูปโภค",
    "ค่าเช่า": "ที่พักและสาธารณูปโภค", "อินเทอร์เน็ต": "ที่พักและสาธารณูปโภค",
    "netflix": "บันเทิงและสมาชิก", "spotify": "บันเทิงและสมาชิก",
    "youtube premium": "บันเทิงและสมาชิก",
    "หนังสือ": "การเรียน", "ค่าเทอม": "การเรียน",
    "bts": "เดินทาง", "mrt": "เดินทาง", "ค่าทางด่วน": "เดินทาง",
    "เงินเดือน": INCOME, "เงินจากผู้ปกครอง": INCOME,
}

SLIP_PROMPT = """อ่านเอกสารนี้ ซึ่งมีสลิปโอนเงินหนึ่งหรือหลายใบ
ตอบเป็น JSON array เท่านั้น แต่ละสลิปเป็น 1 รายการ:
[{"date": "", "merchant": "", "amount": 0, "memo": "", "category": "", "confidence": 0.0}]
date ให้ใช้รูปแบบ YYYY-MM-DD ถ้าในสลิปไม่มีวันที่ให้เว้นว่าง
หมวดที่เลือกได้: """ + ", ".join(CATS) + """
ถ้าไม่แน่ใจให้ใช้ "โอนให้บุคคล/อื่นๆ" และให้ confidence ต่ำ"""

SAMPLE = [
    ("ฟิตเนส", 673.27, "ค่าใช้จ่ายด้านสุขภาพ", "สุขภาพ"),
    ("เงินจากผู้ปกครอง", 8554.00, "รายรับ", "รายรับ"),
    ("โอนให้เพื่อน", 142.78, "โอนเงินค่าใช้จ่ายอื่น ๆ", "โอนให้บุคคล/อื่นๆ"),
    ("Spotify", 343.97, "ความบันเทิงและค่าสมาชิก", "บันเทิงและสมาชิก"),
    ("ปั๊มน้ำมัน", 2204.10, "ค่าเดินทาง", "เดินทาง"),
    ("ค่าไฟ", 3613.65, "ค่าที่พักและสาธารณูปโภค", "ที่พักและสาธารณูปโภค"),
    ("7-Eleven", 85.00, "", "ของใช้และซูเปอร์มาร์เก็ต"),
    ("GrabFood", 185.00, "ค่าอาหาร", "อาหารและเครื่องดื่ม"),
    ("ร้านข้าวมันไก่", 60.00, "ค่าอาหาร", "อาหารและเครื่องดื่ม"),
    ("BTS", 45.00, "ค่าเดินทาง", "เดินทาง"),
    ("Shopee", 899.00, "ช้อปปิ้ง", "ช้อปปิ้ง"),
    ("หนังสือ", 450.00, "ค่าหนังสือเรียน", "การเรียน"),
    ("ค่าเช่า", 4500.00, "ค่าที่พักและสาธารณูปโภค", "ที่พักและสาธารณูปโภค"),
    ("GrabFood", 220.00, "ค่าอาหาร", "อาหารและเครื่องดื่ม"),
    ("Netflix", 419.00, "ความบันเทิงและค่าสมาชิก", "บันเทิงและสมาชิก"),
    ("ร้านยา", 320.00, "ค่าใช้จ่ายด้านสุขภาพ", "สุขภาพ"),
]

# ---------------- ฟังก์ชัน AI ----------------
@st.cache_resource
def get_client():
    key = None
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else None


def ask(contents, tries=3):
    client = get_client()
    if client is None:
        raise RuntimeError("ยังไม่ได้ตั้งค่า GEMINI_API_KEY (ดูที่ Secrets)")
    last = None
    for attempt in range(tries):
        for model in MODELS:
            try:
                return client.models.generate_content(model=model, contents=contents)
            except Exception as e:
                last = e
                if any(x in str(e) for x in ("503", "429", "UNAVAILABLE")):
                    continue
                raise
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"AI ไม่ว่างชั่วคราว ลองใหม่อีกครั้ง ({str(last)[:60]})")


def parse_json(text):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


@st.cache_data(show_spinner=False)
def extract_slips(data: bytes, mime: str):
    chunks = []
    if mime == "application/pdf":
        reader = PdfReader(io.BytesIO(data))
        for s in range(0, min(len(reader.pages), 150), 10):
            w = PdfWriter()
            for p in reader.pages[s:s + 10]:
                w.add_page(p)
            buf = io.BytesIO()
            w.write(buf)
            chunks.append((buf.getvalue(), "application/pdf"))
    else:
        chunks.append((data, mime))
    rows = []
    for b, m in chunks:
        resp = ask([types.Part.from_bytes(data=b, mime_type=m), SLIP_PROMPT])
        rows += parse_json(resp.text)
    return rows


def apply_rules(merchant, memo):
    text = f"{merchant} {memo}".lower()
    for k, v in RULES.items():
        if k in text:
            return v
    return None


def classify_rows(df):
    """จัดหมวดรายการที่ยังไม่มีหมวด: กฎก่อน แล้วค่อยให้ AI"""
    df = df.reset_index(drop=True)
    cats = [apply_rules(m, mm) for m, mm in zip(df["merchant"], df["memo"])]
    conf = [1.0 if c else None for c in cats]
    todo = [i for i, c in enumerate(cats) if c is None]
    for s in range(0, len(todo), 50):
        idx = todo[s:s + 50]
        lines = "\n".join(f"{i}|{df.at[i, 'merchant']}|{df.at[i, 'amount']}|{df.at[i, 'memo']}" for i in idx)
        prompt = (
            "จัดหมวดรายการเงินต่อไปนี้ (รูปแบบ: ลำดับ|ผู้รับหรือแหล่งที่มา|จำนวนเงินบาท|บันทึก)\n"
            f"หมวดที่เลือกได้: {', '.join(CATS)}\n"
            'ถ้าไม่แน่ใจให้ใช้ "โอนให้บุคคล/อื่นๆ" และให้ confidence ต่ำ\n'
            'ตอบเป็น JSON array เท่านั้น: [{"i": 0, "category": "", "confidence": 0.0}]\n'
            f"รายการ:\n{lines}"
        )
        for d in parse_json(ask(prompt).text):
            i = int(d["i"])
            cats[i], conf[i] = d.get("category"), d.get("confidence", 0.5)
    df["category"] = cats
    df["confidence"] = conf
    return df


def normalize(rows):
    df = pd.DataFrame(rows)
    for c in ["date", "merchant", "amount", "memo", "category", "confidence"]:
        if c not in df:
            df[c] = None
    df = df[["date", "merchant", "amount", "memo", "category", "confidence"]].copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs().fillna(0)
    for c in ["date", "merchant", "memo"]:
        df[c] = df[c].fillna("").astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.5)
    df["category"] = df["category"].where(df["category"].isin(CATS), OTHER)
    df["ตรวจสอบ"] = df["confidence"].apply(lambda x: "⚠️" if x < 0.7 else "")
    return df.reset_index(drop=True)


def month_of(d):
    try:
        return pd.to_datetime(d).strftime("%Y-%m")
    except Exception:
        return "ไม่ระบุวันที่"


def set_data(df, seconds=None):
    st.session_state["df"] = df
    st.session_state["seconds"] = seconds
    st.session_state.pop("editor", None)


# ---------------- หน้าเว็บ ----------------
st.title("💰 AI Budget Advisor")
st.caption("อัปโหลดสลิปหรือรายการเดินบัญชี ให้ AI จัดหมวดหมู่ สรุปรายเดือน และเตือนเมื่อใช้เกินงบ")
st.info("🔒 กรุณาอย่าอัปโหลดข้อมูลบัญชีจริง ใช้ข้อมูลตัวอย่างหรือข้อมูลที่ปิดชื่อ/เลขบัญชีแล้ว "
        "ระบบไม่บันทึกข้อมูลของคุณลงฐานข้อมูล (เก็บเฉพาะระหว่างใช้งาน)")

with st.sidebar:
    st.header("ตั้งงบรายหมวด (บาท/เดือน)")
    st.caption("เว้นเป็น 0 = ไม่ตั้งงบ")
    budgets = {c: st.number_input(c, min_value=0, value=0, step=500) for c in EXPENSE_CATS}

tab1, tab2, tab3 = st.tabs(["📷 สลิป", "📄 รายการเดินบัญชี (CSV)", "🧪 ข้อมูลตัวอย่าง"])

with tab1:
    files = st.file_uploader("อัปโหลดรูปสลิป (PNG/JPG) หรือ PDF", type=["png", "jpg", "jpeg", "pdf"],
                             accept_multiple_files=True)
    if st.button("ให้ AI อ่านสลิป", disabled=not files):
        t0, rows = time.time(), []
        try:
            bar = st.progress(0.0, text="กำลังอ่านสลิป...")
            for n, f in enumerate(files[:30], 1):
                rows += extract_slips(f.getvalue(), f.type or "application/pdf")
                bar.progress(n / min(len(files), 30), text=f"อ่านแล้ว {n}/{min(len(files), 30)} ไฟล์")
            df = normalize(rows)
            df["category"] = [apply_rules(m, mm) or c for m, mm, c in zip(df["merchant"], df["memo"], df["category"])]
            set_data(df, time.time() - t0)
        except Exception as e:
            st.error(f"อ่านสลิปไม่สำเร็จ: {e}")

with tab2:
    up = st.file_uploader("อัปโหลดไฟล์ CSV", type=["csv"])
    if up:
        try:
            raw = pd.read_csv(up, encoding="utf-8-sig")
        except Exception:
            up.seek(0)
            raw = pd.read_csv(up)
        if raw.empty:
            st.warning("ไฟล์ไม่มีข้อมูล")
        else:
            cols = list(raw.columns)
            opt = ["(ไม่มี)"] + cols

            def guess(keys, default=0):
                for i, c in enumerate(cols):
                    if any(k in str(c).lower() for k in keys):
                        return i
                return default

            c1, c2, c3, c4 = st.columns(4)
            m_col = c1.selectbox("คอลัมน์ผู้รับ/ร้านค้า", cols, index=guess(["merchant", "ร้าน", "description", "รายละเอียด"]))
            a_col = c2.selectbox("คอลัมน์จำนวนเงิน", cols, index=guess(["amount", "จำนวน", "เงิน"]))
            d_col = c3.selectbox("คอลัมน์วันที่ (ถ้ามี)", opt, index=0)
            n_col = c4.selectbox("คอลัมน์บันทึก (ถ้ามี)", opt, index=0)
            if st.button("ให้ AI จัดหมวด"):
                t0 = time.time()
                try:
                    base = pd.DataFrame({
                        "date": raw[d_col] if d_col != "(ไม่มี)" else "",
                        "merchant": raw[m_col],
                        "amount": pd.to_numeric(raw[a_col], errors="coerce").abs().fillna(0),
                        "memo": raw[n_col] if n_col != "(ไม่มี)" else "",
                    }).head(300)
                    base["merchant"] = base["merchant"].fillna("").astype(str)
                    base["memo"] = base["memo"].fillna("").astype(str)
                    with st.spinner("AI กำลังจัดหมวด..."):
                        out = classify_rows(base)
                    set_data(normalize(out.to_dict("records")), time.time() - t0)
                except Exception as e:
                    st.error(f"จัดหมวดไม่สำเร็จ: {e}")
        st.caption("อ่านสูงสุด 300 แถวต่อครั้ง")

with tab3:
    st.write("ไม่มีไฟล์? กดโหลดข้อมูลตัวอย่าง (ไม่ต้องใช้ AI)")
    if st.button("โหลดข้อมูลตัวอย่าง"):
        rows = [{"date": "", "merchant": m, "amount": a, "memo": mm, "category": c, "confidence": 1.0}
                for m, a, mm, c in SAMPLE]
        set_data(normalize(rows))

# ---------------- ผลลัพธ์ ----------------
if "df" in st.session_state:
    st.divider()
    st.subheader("1) ตรวจและแก้ไขหมวดหมู่")
    if st.session_state.get("seconds"):
        st.caption(f"⏱️ ประมวลผล {len(st.session_state['df'])} รายการ ใน {st.session_state['seconds']:.1f} วินาที")
    st.caption("รายการที่มี ⚠️ คือ AI ไม่มั่นใจ ควรตรวจก่อน แก้หมวดในตารางได้โดยตรง")
    edited = st.data_editor(
        st.session_state["df"], key="editor", use_container_width=True, hide_index=True,
        disabled=["confidence", "ตรวจสอบ"],
        column_config={"category": st.column_config.SelectboxColumn("หมวดหมู่", options=CATS, required=True)},
    )

    view = edited.copy()
    view["เดือน"] = view["date"].map(month_of)
    months = sorted(view["เดือน"].unique())
    month = st.selectbox("เดือนที่ดู", months) if len(months) > 1 else months[0]
    view = view[view["เดือน"] == month]

    income = view.loc[view["category"] == INCOME, "amount"].sum()
    exp = view[view["category"] != INCOME]
    spent = exp.groupby("category")["amount"].sum().sort_values(ascending=False)

    st.subheader("2) สรุป")
    k1, k2, k3 = st.columns(3)
    k1.metric("รายรับ", f"{income:,.0f} บาท")
    k2.metric("รายจ่าย", f"{spent.sum():,.0f} บาท")
    k3.metric("คงเหลือ", f"{income - spent.sum():,.0f} บาท")
    if not spent.empty:
        st.bar_chart(spent)

    st.subheader("3) งบและการแจ้งเตือน")
    any_budget = False
    for cat in EXPENSE_CATS:
        b, s = budgets[cat], float(spent.get(cat, 0))
        if b > 0:
            any_budget = True
            st.progress(min(s / b, 1.0), text=f"{cat}: {s:,.0f} / {b:,.0f} บาท")
            if s > b:
                st.error(f"⚠️ {cat} เกินงบ {s - b:,.0f} บาท")
    if not any_budget:
        st.info("ตั้งงบรายหมวดที่แถบด้านซ้าย เพื่อให้ระบบเตือนเมื่อใช้เกินงบ")

    st.subheader("4) คำแนะนำการออมจาก AI")
    if st.button("ขอคำแนะนำ", disabled=spent.empty):
        over = {c: float(spent.get(c, 0)) - b for c, b in budgets.items() if b > 0 and float(spent.get(c, 0)) > b}
        prompt = (
            f"รายรับ {income:,.0f} บาท รายจ่ายแยกหมวด: {spent.round(0).to_dict()}\n"
            f"หมวดที่เกินงบ (เกินกี่บาท): {over or 'ไม่มี'}\n"
            "ให้คำแนะนำการออมแบบเป็นกันเอง 3-4 ข้อ สั้น กระชับ ทำได้จริงสำหรับนักศึกษา/คนทำงานรุ่นใหม่ "
            "อ้างอิงเฉพาะตัวเลขที่ให้มา ห้ามแต่งตัวเลขเพิ่ม"
        )
        try:
            with st.spinner("AI กำลังวิเคราะห์..."):
                st.write(ask(prompt).text)
            st.caption("เป็นคำแนะนำทั่วไป ไม่ใช่คำปรึกษาทางการเงินเชิงวิชาชีพ")
        except Exception as e:
            st.error(f"ขอคำแนะนำไม่สำเร็จ: {e}")

    st.download_button("ดาวน์โหลดผลเป็น CSV", edited.to_csv(index=False, encoding="utf-8-sig"),
                       file_name="budget_result.csv", mime="text/csv")

st.divider()
st.subheader("แบบประเมินการใช้งาน")
if FEEDBACK_URL:
    st.link_button("ตอบแบบประเมิน (1 นาที)", FEEDBACK_URL)
else:
    st.caption("(ผู้พัฒนา: ใส่ลิงก์ Google Form ที่ตัวแปร FEEDBACK_URL ในไฟล์ app.py)")

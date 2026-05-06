import streamlit as st
from faster_whisper import WhisperModel
from phonemizer import phonemize
import difflib
import re

# モデルの読み込み（初回実行時に自動ダウンロードされます）
# "base"は軽量で速いモデルです。精度を上げるなら"small"や"medium"に変更可
@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

def get_ipa(text):
    """テキストを音素(IPA)に変換"""
    return phonemize(text, language='en-us', backend='espeak')

def analyze_diff(script, transcript):
    """間違いを比較し、赤字で表示"""
    s_words = re.sub(r'[.,!?]', '', script.lower()).split()
    t_words = re.sub(r'[.,!?]', '', transcript.lower()).split()
    
    diff = difflib.ndiff(s_words, t_words)
    result_display = []
    error_count = 0
    errors = []

    for word in diff:
        if word.startswith('  '):
            result_display.append(word[2:])
        elif word.startswith('- '): # 台本にあるが読めなかった
            error_count += 1
            w = word[2:]
            result_display.append(f"**:red[{w}]**")
            errors.append(w)
    
    return " ".join(result_display), error_count, errors

# --- UI ---
st.title("🎤 完全無料・発音矯正アプリ")
st.caption("OpenAIを使わず、あなたのPC内で全て処理します（プライバシー安全）")

col1, col2 = st.columns(2)
with col1:
    script_input = st.text_area("1. 台本を入力", "Check it out at the end of the day.")
with col2:
    audio_file = st.file_uploader("2. 音声ファイルをアップロード", type=['mp3', 'wav', 'm4a'])

if st.button("分析を実行"):
    if audio_file and script_input:
        with st.spinner("PC内で音声解析中..."):
            # 一時ファイルとして保存して読み込み
            with open("temp_audio.mp3", "wb") as f:
                f.write(audio_file.read())
            
            segments, info = model.transcribe("temp_audio.mp3", beam_size=5)
            transcript_text = "".join([segment.text for segment in segments])
            # 確信度（プロバビリティ）の平均を計算
            confidence = info.language_probability * 100

        # 分析
        display_text, err_count, err_list = analyze_diff(script_input, transcript_text)

        st.subheader("分析結果")
        st.markdown(f"**添削結果:** {display_text}")
        st.write(f"❌ 間違い箇所: {err_count}箇所")
        st.info(f"💡 発音の明瞭度スコア: {confidence:.1f}%")

        # 音素分析
        if err_list:
            st.subheader("🔤 苦手な音の分析")
            for err in set(err_list):
                ipa = get_ipa(err).strip()
                st.write(f"・**{err}** [/{ipa}/]")
                if 'r' in ipa: st.caption("→ /r/ の舌の動きを意識しましょう。")
                if 'l' in ipa: st.caption("→ /l/ の弾きを強くしましょう。")
    else:
        st.error("台本と音声の両方を準備してください。")

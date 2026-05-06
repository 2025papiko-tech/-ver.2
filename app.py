import streamlit as st
from faster_whisper import WhisperModel
from phonemizer import phonemize
import difflib
import re
from collections import defaultdict

# モデルの読み込み
@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

def get_ipa(text):
    """テキストを音素(IPA)に変換"""
    return phonemize(text, language='en-us', backend='espeak')

def analyze_diff(script, transcript):
    """間違いを比較し、赤字で表示"""
    s_words = re.sub(r'[.,!?—]', '', script.lower()).split()
    t_words = re.sub(r'[.,!?—]', '', transcript.lower()).split()
    
    diff = difflib.ndiff(s_words, t_words)
    result_display = []
    error_count = 0
    errors = []

    for word in diff:
        if word.startswith('  '):
            result_display.append(word[2:])
        elif word.startswith('- '):
            error_count += 1
            w = word[2:]
            result_display.append(f"**:red[{w}]**")
            errors.append(w)
    
    return " ".join(result_display), error_count, errors

def extract_audio_changes(text):
    """音声変化（リダクション・フラップT等）の抽出"""
    text = text.lower()
    changes = []
    
    # 1. 簡略化・リダクションのパターン
    patterns = {
        r"going to": "gonna (going toはgonnaと発音されやすい)",
        r"want to": "wanna (want toはwannaと発音されやすい)",
        r"got to": "gotta (got toはgottaと発音されやすい)",
        r"kind of": "kinda (kind ofはkindaと発音されやすい)",
    }
    for p, desc in patterns.items():
        if re.search(p, text):
            changes.append(f"✅ **Reduction**: {desc}")

    # 2. フラップT/D (母音に挟まれたT)
    # 簡易的にbetter, water, started, excitingなどを抽出
    flap_matches = re.findall(r'\b\w+[aeiou]t[aeiou]\w*\b', text)
    if flap_matches:
        unique_flaps = set(flap_matches)
        changes.append(f"✅ **Flap T/D**: {', '.join(list(unique_flaps)[:3])}などは、Tの音が濁って『d』や『l』に近い音になります(better→beder現象)。")

    # 3. リンキング (子音 + 母音)
    linking_matches = re.findall(r'\b\w*[^aeiou] [aeiou]\w*\b', text)
    if linking_matches:
        changes.append(f"✅ **Linking**: {' / '.join(linking_matches[:2])}などは、前の単語の最後と次の単語の頭を繋げて発音しましょう。")

    return changes

# --- UI ---
st.set_page_config(page_title="English Pronunciation Coach", page_icon="🎤")
st.title("🎤 AI英語発音・暗唱矯正アプリ")

script_input = st.text_area("1. 正解の台本を入力", height=200)
audio_file = st.file_uploader("2. 録音ファイルをアップロード", type=['mp3', 'wav', 'm4a'])

if st.button("分析を実行"):
    if audio_file and script_input:
        with st.spinner("音声を精密解析中..."):
            with open("temp_audio.mp3", "wb") as f:
                f.write(audio_file.read())
            
            segments, info = model.transcribe("temp_audio.mp3", beam_size=5)
            transcript_text = " ".join([segment.text for segment in segments])
            confidence = info.language_probability * 100

        display_text, err_count, err_list = analyze_diff(script_input, transcript_text)

        st.subheader("📊 分析結果")
        st.markdown(f"**添削結果 (赤字がミス):**\n\n{display_text}")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("間違い箇所", f"{err_count} 単語")
        col_res2.metric("発音明瞭度", f"{confidence:.1f}%")

        # --- 音声変化セクション ---
        st.subheader("🔗 意識すべき音声変化")
        audio_changes = extract_audio_changes(script_input)
        if audio_changes:
            for change in audio_changes:
                st.write(change)
        else:
            st.write("特に顕著な音声変化パターンは見つかりませんでしたが、単語同士を滑らかに繋げる意識を持ちましょう。")

        # --- 音素パターン別セクション ---
        if err_list:
            st.subheader("🔤 苦手な音のパターン別解説")
            st.write("ミスした単語を原因別にまとめました。優先的に復習しましょう。")
            
            pattern_groups = defaultdict(list)
            
            for err in set(err_list):
                ipa = get_ipa(err).strip()
                item = f"{err} [/{ipa}/]"
                
                if 'l' in ipa:
                    pattern_groups['Lの音（弾く音）'].append(item)
                if 'ɹ' in ipa or 'r' in ipa:
                    pattern_groups['Rの音（こもる音）'].append(item)
                if 'ð' in ipa or 'θ' in ipa:
                    pattern_groups['THの音（摩擦音）'].append(item)
                if 'v' in ipa or 'f' in ipa:
                    pattern_groups['V/Fの音（唇を噛む音）'].append(item)
                if 'æ' in ipa:
                    pattern_groups['アとエの中間音（æ）'].append(item)

            # カテゴリごとに表示
            for category, words in pattern_groups.items():
                with st.expander(f"📌 {category} - {len(words)}単語"):
                    if 'L' in category:
                        st.success("💡 **解説**: 舌先を前歯の付け根に強く押し当て、離す瞬間に「パッ」と弾きます。")
                    elif 'R' in category:
                        st.warning("💡 **解説**: 舌をどこにも触れさせず、奥に引いて喉を鳴らすイメージです。")
                    elif 'TH' in category:
                        st.info("💡 **解説**: 舌先を前歯で軽く挟むか、隙間から息を強く漏らします。")
                    elif 'V' in category:
                        st.info("💡 **解説**: 上の前歯で下唇を軽く押さえ、振動させながら息を出します。")
                    elif 'æ' in category:
                        st.info("💡 **解説**: 指二本分くらい口を縦横に大きく開け、アとエを同時に言う音です。")
                    
                    st.write("対象単語: " + " / ".join(words))
    else:
        st.error("台本と音声ファイルを両方準備してください。")

st.divider()
st.caption("Pro-tip: 好きなバンドの歌詞を台本にすると、より感情を乗せた発音練習ができます。")

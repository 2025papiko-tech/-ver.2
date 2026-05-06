import streamlit as st
from faster_whisper import WhisperModel
from phonemizer import phonemize
import difflib
import re

# モデルの読み込み（初回実行時に自動ダウンロードされます）
@st.cache_resource
def load_model():
    # "base"は軽量モデル。より高精度が必要なら"small"に変更可能
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

def get_ipa(text):
    """テキストを音素(IPA)に変換"""
    return phonemize(text, language='en-us', backend='espeak')

def analyze_diff(script, transcript):
    """間違いを比較し、赤字で表示"""
    # 記号を除去して小文字化
    s_words = re.sub(r'[.,!?—]', '', script.lower()).split()
    t_words = re.sub(r'[.,!?—]', '', transcript.lower()).split()
    
    diff = difflib.ndiff(s_words, t_words)
    result_display = []
    error_count = 0
    errors = []

    for word in diff:
        if word.startswith('  '):
            result_display.append(word[2:])
        elif word.startswith('- '): # 台本にあるが読めなかった（またはミスした）単語
            error_count += 1
            w = word[2:]
            result_display.append(f"**:red[{w}]**")
            errors.append(w)
    
    return " ".join(result_display), error_count, errors

# --- UI ---
st.set_page_config(page_title="English Pronunciation AI", page_icon="🎤")
st.title("🎤 AI英語発音・暗唱矯正アプリ")
st.caption("OpenAI不要・完全無料。あなたのPC/サーバー内で解析します。")

col1, col2 = st.columns(2)
with col1:
    script_input = st.text_area("1. 正解の台本を入力", "Check it out at the end of the day.")
with col2:
    audio_file = st.file_uploader("2. 録音ファイルをアップロード", type=['mp3', 'wav', 'm4a'])

if st.button("分析を実行"):
    if audio_file and script_input:
        with st.spinner("音声を解析中..."):
            # 一時ファイルとして保存
            with open("temp_audio.mp3", "wb") as f:
                f.write(audio_file.read())
            
            segments, info = model.transcribe("temp_audio.mp3", beam_size=5)
            transcript_text = " ".join([segment.text for segment in segments])
            confidence = info.language_probability * 100

        # 分析実行
        display_text, err_count, err_list = analyze_diff(script_input, transcript_text)

        st.subheader("📊 分析結果")
        st.markdown(f"**添削結果 (赤字がミス):**\n\n{display_text}")
        
        st.divider()
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("間違い箇所", f"{err_count} 単語")
        col_res2.metric("発音明瞭度", f"{confidence:.1f}%")

        # 音声変化アドバイス
        st.subheader("🔗 音声変化(Linking/Reduction)のコツ")
        tips = []
        if "it out" in script_input.lower(): tips.append("・**it out**: 't'がラ行化して『イラウト』のように繋がります(Flap T)。")
        if "at the" in script_input.lower(): tips.append("・**at the**: 't'を飲み込み『アッダ』のように発音するとスムーズです。")
        if "check it" in script_input.lower(): tips.append("・**check it**: 'k'と'i'が繋がって『チェキッ』になります(Linking)。")
        
        if tips:
            for tip in tips: st.write(tip)
        else:
            st.write("単語同士を繋げる意識で話すとより自然になります。")

        # 音素分析の改善
        if err_list:
            st.subheader("🔤 苦手な音の集中解説")
            st.write("ミスした単語に含まれる、注意すべき音素です。")
            
            # 重複を除去して分析
            for err in sorted(set(err_list)):
                ipa = get_ipa(err).strip()
                st.write(f"---")
                st.write(f"単語: **{err}** `/{ipa}/`")
                
                # 具体的な発音アドバイスの分岐
                if 'l' in ipa:
                    st.success("💡 **Lの音**: 舌先を上の前歯の付け根にしっかり押し当てて、パッと「弾く」ように離して！")
                
                if 'ɹ' in ipa or 'r' in ipa:
                    st.warning("💡 **Rの音**: 舌をどこにも付けず、喉の奥に引いて「ウー」に近い音でこもらせて。")
                
                if 'ð' in ipa or 'θ' in ipa:
                    st.info("💡 **THの音**: 舌先を前歯で軽く挟むか、裏側に触れさせて空気を漏らして。")
                
                if 'v' in ipa:
                    st.info("💡 **Vの音**: 上の前歯で下唇を軽く押さえて、震わせながら音を出して。")
                
                if 'æ' in ipa:
                    st.info("💡 **エとアの中間音**: 口を左右に大きく開けて「ア」と言うつもりで「エ」を出してみて。")

    else:
        st.error("台本と音声ファイルを両方準備してください。")

st.divider()
st.caption("Tips: THE BAWDIESのようなガレージロックを歌う際は、'R'や'V'の音を強調するとよりエネルギッシュに聞こえます！")

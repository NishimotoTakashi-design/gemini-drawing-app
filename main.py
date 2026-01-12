import streamlit as st
import google.generativeai as genai
import json
from PIL import Image
import io

# ==========================================
# 1. セキュリティ設定（パスワード & APIキー）
# ==========================================
def check_password():
    """簡易パスワード認証（公開範囲のコントロール）"""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        password = st.sidebar.text_input("パスワードを入力してください", type="password")
        if password == st.secrets.get("APP_PASSWORD", "admin123"): # Secretsで設定
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.warning("パスワードが正しくありません")
            return False
    return True

# APIキーの設定（Streamlit CloudのSecretsに設定しておく）
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("APIキーが設定されていません。")

# ==========================================
# 2. メインロジック（図面解析）
# ==========================================
def analyze_drawing(image, target_columns):
    """Geminiを使用して図面から情報を抽出する"""
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # 抽出したいカラムをプロンプトに組み込む
    prompt = f"""
    この図面から以下の情報を抽出して、JSON形式で出力してください。
    抽出項目: {target_columns}
    
    出力フォーマット例:
    {{
        "項目名1": "値1",
        "項目名2": "値2"
    }}
    """
    
    response = model.generate_content([prompt, image])
    return response.text

# ==========================================
# 3. UI 構築
# ==========================================
st.set_page_config(page_title="図面情報構造化ツール", layout="wide")

if check_password():
    st.title("📄 図面情報 構造化ツール")
    st.write("図面から特定の情報を抽出し、構造化データ（JSON/表形式）に変換します。")

    # サイドバー：設定
    st.sidebar.header("設定")
    input_method = st.sidebar.radio("インプット方法を選択", ("ローカルからアップロード", "Google Driveパス指定"))
    
    target_columns = st.sidebar.text_area(
        "抽出するカラムを指定（カンマ区切り）",
        "図番, 品名, 材質, 表面処理, 最大寸法, メーカー"
    )

    # メインエリア：ファイル入力
    img_content = None
    
    if input_method == "ローカルからアップロード":
        uploaded_file = st.file_uploader("図面（画像/PDF）をアップロードしてください", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            img_content = Image.open(uploaded_file)
            st.image(img_content, caption="アップロードされた図面", width=400)

    else:
        drive_path = st.text_input("Google DriveのフォルダパスまたはファイルIDを入力してください")
        st.info("※Google Drive連携には、別途Google Drive APIの認証(Service Account等)が必要です。")
        # ここにGoogle DriveからファイルをDLする関数を呼び出す処理を記述

    # 解析実行
    if st.button("構造化を実行する") and img_content:
        with st.spinner("解析中..."):
            try:
                result_text = analyze_drawing(img_content, target_columns)
                
                # 結果表示
                st.subheader("解析結果")
                st.code(result_text, language='json')
                
                # JSONとしてパースできれば表形式で表示
                # (マークダウン内のJSONを抽出する処理が必要な場合があります)
                try:
                    # 前後のマークダウン（```json ... ```）を削除してパース
                    clean_json = result_text.strip().replace("```json", "").replace("```", "")
                    data_dict = json.loads(clean_json)
                    st.table([data_dict])
                except:
                    st.warning("解析結果を表形式に変換できませんでした。テキストを確認してください。")
                    
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import json
import io
import plotly.graph_objs as go
import plotly.express as px
import pytz
import requests
import google.generativeai as genai
from datetime import date, timedelta
from fpdf import FPDF
import streamlit.components.v1 as components
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import unicodedata

# ---- Session Initialization ----
if "email" not in st.session_state:
    st.session_state.email = ""
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'history' not in st.session_state:
    st.session_state.history = []

st.title(" 🪐 OrbitTrader AI(AI Trading Simulator)")

# ---- Sidebar Input Section ----
st.sidebar.header("Input / Upload")

st.sidebar.subheader("Add Manual Trade")
ticker = st.sidebar.text_input("Ticker (e.g., AAPL)").upper()
shares = st.sidebar.number_input("Shares", min_value=1, step=1)
trade_type = st.sidebar.radio("Trade Type", ['Buy', 'Sell'])
trade_date = st.sidebar.date_input("Trade Date", value=datetime.date.today())
submit = st.sidebar.button("Submit Trade")

# ---- JSON Import ----
st.sidebar.subheader("Import from JSON")
json_file = st.sidebar.file_uploader("Upload JSON file", type=["json"])
if json_file is not None:
    try:
        content = json.load(json_file)
        for trade in content:
            if trade['Trade'] == 'Buy':
                st.session_state.portfolio.append({
                    'Ticker': trade['Ticker'],
                    'Shares': trade['Shares'],
                    'Buy Price': trade['Price'],
                    'Sector': trade.get('Sector', 'Unknown'),
                    'Date': trade['Date']
                })
            st.session_state.history.append(trade)
        st.sidebar.success("Trades imported successfully.")
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")

# ---- JSON Export ----
st.sidebar.subheader("Export to JSON")
if st.sidebar.button("Download Trades as JSON"):
    export_data = st.session_state.history
    json_bytes = io.BytesIO()
    json_bytes.write(json.dumps(export_data, indent=2).encode('utf-8'))
    json_bytes.seek(0)
    st.sidebar.download_button("Click to Download", data=json_bytes, file_name="trades_export.json")

# ---- CSV Upload & Convert to JSON ----
st.sidebar.subheader("Upload Yahoo CSV and Convert")
uploaded_csv = st.sidebar.file_uploader("Upload Yahoo Finance CSV", type=["csv"])

def get_sector(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get("sector", "Unknown")
    except:
        return "Unknown"

def convert_yahoo_df_to_trades(df):
    try:
        df = df.dropna(subset=["Trade Date"]).copy()
        df["Trade Date"] = df["Trade Date"].astype(int).astype(str)
        df["Trade Date"] = pd.to_datetime(df["Trade Date"], format="%Y%m%d").dt.date
        unique_tickers = df["Symbol"].unique()
        sector_map = {ticker: get_sector(ticker) for ticker in unique_tickers}
        df["Sector"] = df["Symbol"].map(sector_map)

        trades = []
        for _, row in df.iterrows():
            trades.append({
                "Date": str(row["Trade Date"]),
                "Ticker": row["Symbol"],
                "Trade": "Buy",
                "Shares": float(row["Quantity"]),
                "Price": float(row["Purchase Price"]),
                "Sector": row["Sector"]
            })
        return trades
    except Exception as e:
        st.sidebar.error(f"❌ Conversion failed: {e}")
        return []

if uploaded_csv is not None:
    try:
        df_csv = pd.read_csv(uploaded_csv)
        trades_from_csv = convert_yahoo_df_to_trades(df_csv)
        for trade in trades_from_csv:
            if trade['Trade'] == 'Buy':
                st.session_state.portfolio.append({
                    'Ticker': trade['Ticker'],
                    'Shares': trade['Shares'],
                    'Buy Price': trade['Price'],
                    'Sector': trade.get('Sector', 'Unknown'),
                    'Date': trade['Date']
                })
            st.session_state.history.append(trade)
        st.sidebar.success("✅ CSV converted and loaded into portfolio!")
    except Exception as e:
        st.sidebar.error(f"❌ Upload failed: {e}")

# ---- Email Popover (at bottom of sidebar)
with st.sidebar.popover("📧 Enter your email for updates"):
    email_input = st.text_input("Email Address")
    if st.button("Save Email"):
        st.session_state.email = email_input
        st.success("Email saved!")

# ---- Manual Trade Logic ----
if submit and ticker:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=trade_date - datetime.timedelta(days=7), end=trade_date + datetime.timedelta(days=1))
        hist = hist.sort_index()
        hist.index = hist.index.tz_localize(None)
        hist = hist.asof(pd.to_datetime(str(trade_date)))
        price = hist['Close'] if not hist.empty else stock.info['previousClose']
        sector = stock.info.get('sector', 'Unknown')

        if trade_type == 'Buy':
            st.session_state.portfolio.append({
                'Ticker': ticker,
                'Shares': shares,
                'Buy Price': price,
                'Sector': sector,
                'Date': str(trade_date)
            })
            st.sidebar.success(f"Bought {shares} shares of {ticker} at ${price:.2f} on {trade_date}")

        elif trade_type == 'Sell':
            total_sold = 0
            for trade in st.session_state.portfolio:
                if trade['Ticker'] == ticker and trade['Shares'] > 0:
                    to_sell = min(trade['Shares'], shares - total_sold)
                    trade['Shares'] -= to_sell
                    total_sold += to_sell
                    if total_sold >= shares:
                        break
            st.sidebar.success(f"Sold {shares} shares of {ticker} on {trade_date}")

        st.session_state.history.append({
            'Date': str(trade_date),
            'Ticker': ticker,
            'Trade': trade_type,
            'Shares': shares,
            'Price': price,
            'Sector': sector
        })

    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ---- Portfolio Table ----
st.subheader("Current Portfolio")
if st.session_state.portfolio:
    df = pd.DataFrame(st.session_state.portfolio)
    df = df.groupby(['Ticker', 'Sector']).agg({'Shares': 'sum', 'Buy Price': 'mean'}).reset_index()
    current_prices = []
    for ticker in df['Ticker']:
        try:
            current_prices.append(yf.Ticker(ticker).info['regularMarketPrice'])
        except:
            current_prices.append(0)
    df['Current Price'] = current_prices
    df['Current Value'] = df['Current Price'] * df['Shares']
    df['Total Cost'] = df['Buy Price'] * df['Shares']
    df['P/L ($)'] = df['Current Value'] - df['Total Cost']
    df['P/L (%)'] = ((df['Current Value'] - df['Total Cost']) / df['Total Cost']) * 100
    st.dataframe(df, use_container_width=True)

    # ---- Sector Pie Chart ----
    st.subheader("Allocation by Sector")
    sector_data = df.groupby("Sector")['Current Value'].sum()
    st.plotly_chart(go.Figure(go.Pie(labels=sector_data.index, values=sector_data.values, hole=0.3, title="Sector Breakdown")))
else:
    st.info("No active positions in portfolio.")

# ---- Portfolio Growth ----
if st.session_state.portfolio:
    st.subheader("Simulated Portfolio Growth Over Time")
    all_dates = pd.date_range(
        start=min(pd.to_datetime([t['Date'] for t in st.session_state.portfolio])),
        end=datetime.date.today()
    )
    timeline_df = pd.DataFrame(index=all_dates)
    sector_map = {}
    for trade in st.session_state.portfolio:
        ticker = trade['Ticker']
        shares = trade['Shares']
        sector = trade.get('Sector', 'Unknown')
        sector_map[ticker] = sector
        buy_date = pd.to_datetime(trade['Date'])
        try:
            stock_data = yf.Ticker(ticker).history(start=buy_date, end=datetime.date.today())
            stock_data.index = stock_data.index.tz_localize(None)
            stock_data = stock_data[['Close']].rename(columns={'Close': ticker})
            stock_data[ticker] *= shares
            timeline_df = timeline_df.join(stock_data, how='outer')
        except:
            continue
    timeline_df.ffill(inplace=True)
    timeline_df.fillna(0, inplace=True)
    timeline_df['Total Value'] = timeline_df.sum(axis=1)

    traces = []
    sector_colors = {}
    color_palette = px.colors.qualitative.Set1
    for i, (ticker, sector) in enumerate(sector_map.items()):
        if ticker in timeline_df:
            color = sector_colors.setdefault(sector, color_palette[i % len(color_palette)])
            traces.append(go.Scatter(x=timeline_df.index, y=timeline_df[ticker], mode='lines', name=f"{ticker} ({sector})", line=dict(color=color)))

    traces.append(go.Scatter(x=timeline_df.index, y=timeline_df['Total Value'], mode='lines', name='Total Portfolio', line=dict(color='black', width=3)))
    fig = go.Figure(data=traces)
    fig.update_layout(title="Portfolio Growth by Sector", xaxis_title="Date", yaxis_title="Value ($)")
    st.plotly_chart(fig, use_container_width=True)

# ---- Trade History ----
st.subheader("Trade History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df.sort_values(by="Date", ascending=False), use_container_width=True)
else:
    st.info("No trades made yet.")

# --- API Keys & Analysis Setup ---
genai.configure(api_key="AIzaSyAZA2aZfWN-P6-3w6Oq7QOMGh99bxswD3o")
model = genai.GenerativeModel("gemini-1.5-pro")
NEWS_API_KEY = "c7a2cdbd13d440839be331c12ddaef91"

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_stock_data(ticker):
    end = date.today()
    start = end - timedelta(days=180)
    df = yf.download(ticker, start=start, end=end)
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    return df

def fetch_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&pageSize=5"
    try:
        r = requests.get(url)
        data = r.json()
        return [f"- {a['title']} ({a['source']['name']})" for a in data.get('articles', [])]
    except:
        return []

def build_prompt(ticker, df, news_list):
    recent = df.dropna().iloc[-1]
    close = round(recent['Close'], 2)
    rsi = round(recent['RSI'], 2)
    macd = round(recent['MACD'], 2)
    signal = round(recent['Signal'], 2)
    news_str = "\n".join(news_list) if news_list else "No recent news found."

    return f"""
You are a financial research AI.

Analyze the stock **{ticker}** using the following data:
- Current Price: ${close}
- RSI (14): {rsi}
- MACD: {macd}
- Signal Line: {signal}

Recent News:
{news_str}

Provide a professional 5-paragraph analysis with:
1. Interpretation of technical indicators
2. Market sentiment based on recent news
3. Risk and uncertainty evaluation
4. Short-term trading outlook (Bullish, Bearish, Neutral)
5. Final recommendation

📊 At the end, include:
- 🔮 Market Outlook: [Bullish/Bearish/Neutral]
- 💡 Suggested Action: [HUGE BUY / BUY / HOLD / SELL]
"""

def generate_stock_research(ticker):
    df = get_stock_data(ticker)
    news = fetch_news(ticker)
    prompt = build_prompt(ticker, df, news)
    response = model.generate_content(prompt)
    return response.text

# ---- PDF Builder ----
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "AI Stock Research Report", ln=True, align="C")
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
    def add_report(self, ticker, content):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, f"{ticker} Report", ln=True)
        self.ln(2)
        self.set_font("Arial", "", 12)
        safe_content = unicodedata.normalize('NFKD', content).encode('ascii', 'ignore').decode('ascii')
        for line in safe_content.split("\n"):
            self.multi_cell(0, 10, line)
        self.ln(5)

# ---- Generate PDF and Email
st.subheader("📄 Generate AI Stock Research PDF")
generate_pdf = st.button("Generate Research PDF")
ticker_list = sorted({trade['Ticker'] for trade in st.session_state.portfolio})

if generate_pdf and ticker_list:
    try:
        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        for ticker in ticker_list:
            with st.spinner(f"Generating report for {ticker}..."):
                try:
                    report = generate_stock_research(ticker)
                    pdf.add_report(ticker, report)
                except Exception as e:
                    pdf.add_report(ticker, f"Error generating report: {e}")

        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_buffer = io.BytesIO(pdf_bytes)

        st.success("✅ PDF report generated successfully!")
        st.download_button(
            label="📥 Download AI Stock Research PDF",
            data=pdf_buffer,
            file_name="AI_Stock_Research_Report.pdf",
            mime="application/pdf"
        )

        if st.session_state.email:
            try:
                sender_email = "orbittraderai@gmail.com"
                receiver_email = st.session_state.email
                password = "sjakhldknitnyasz"

                subject = "📄 Your AI Stock Research PDF is Ready!"
                body = """Hello,

Your AI stock research report has been generated successfully.

You can download it by revisiting your Streamlit app session and clicking the '📥 Download AI Stock Research PDF' button.

Or you could download it here

Thank you,
OrbitTrader AI"""

                msg = MIMEMultipart()
                msg["From"] = sender_email
                msg["To"] = receiver_email
                msg["Subject"] = subject
                from email.mime.base import MIMEBase
                from email import encoders

                msg.attach(MIMEText(body, "plain"))

                # Attach PDF
                part = MIMEBase("application", "octet-stream")
                part.set_payload(pdf_bytes)
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename=AI_Stock_Research_Report.pdf")
                msg.attach(part)

                # Send the email
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(sender_email, password)
                    server.sendmail(sender_email, receiver_email, msg.as_string())


                st.info(f"📤 Email sent to {receiver_email}")
            except Exception as e:
                st.warning(f"❌ Email failed: {e}")

    except Exception as e:
        st.error(f"PDF generation failed: {e}")

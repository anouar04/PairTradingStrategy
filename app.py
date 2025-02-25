""""from flask import Flask, render_template, request, jsonify
import ccxt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import time
import matplotlib.pyplot as plt
import base64
import io
import plotly.graph_objs as go
import os
# Set Matplotlib to use a non-GUI backend
plt.switch_backend('Agg')

app = Flask(__name__)

# Initialize GATE.io futures exchange
exchange = ccxt.gateio({
    'options': {
        'defaultType': 'future',  # Use the futures API
    }
})

# Function to generate the Z-Score Plotly chart with zoom and pan features
def generate_zscore_plot(symbol1, symbol2, zscore, days):
    # Create the Z-Score plot using Plotly
    trace = go.Scatter(
        x=zscore.index,
        y=zscore,
        mode='lines',
        name='Z-Score',
        line=dict(color='blue', width=1.5),
        hoverinfo='x+y',  # Show the x and y values when hovering
    )

    # Create threshold lines for the plot
    upper_threshold = go.Scatter(
        x=zscore.index,
        y=[2] * len(zscore),
        mode='lines',
        name='Upper Threshold (2)',
        line=dict(color='red', dash='dash'),
        showlegend=False,  # Do not display in legend
    )

    lower_threshold = go.Scatter(
        x=zscore.index,
        y=[-2] * len(zscore),
        mode='lines',
        name='Lower Threshold (-2)',
        line=dict(color='green', dash='dash'),
        showlegend=False,  # Do not display in legend
    )

    mean_line = go.Scatter(
        x=zscore.index,
        y=[0] * len(zscore),
        mode='lines',
        name='Mean (0)',
        line=dict(color='black', dash='dash'),
        showlegend=False,  # Do not display in legend
    )

    # Combine traces into a figure
    fig = go.Figure(data=[trace, upper_threshold, lower_threshold, mean_line])

    # Update layout with title, axis labels, etc.
    fig.update_layout(
        title=f"Z-Score of the Spread ({symbol1} - {symbol2}) - Last {days} Days",
        xaxis_title="Date and Time",
        yaxis_title="Z-Score",
        xaxis=dict(
            tickformat="%Y-%m-%d %H:%M",
            showgrid=True,
        ),
        yaxis=dict(
            showgrid=True,
        ),
        hovermode="closest",  # Tooltip will show when hovering over the points
        dragmode='zoom',  # Enable zoom and pan mode (similar to TradingView)
        template="plotly_white",  # White theme for the plot
        autosize=True,  # Automatically resize the chart
        margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins for better readability
        height=800,  # Increase the height for a larger chart
        width=1200,  # Increase the width for a larger chart
    )

    # Return the plot HTML div for embedding in the template
    return fig.to_html(full_html=False)

@app.route('/', methods=['GET', 'POST'])
def home():
    # Default values
    symbol1 = request.args.get('symbol1', 'DOGE/USDT')
    symbol2 = request.args.get('symbol2', 'XLM/USDT')
    timeframe = request.args.get('timeframe', '1m')  # Get timeframe from query parameters
    days = int(request.args.get('days', 5))  # Get days from query parameters

    # Calculate the start and end times
    end_time = int(time.time() * 1000)  # Current time in milliseconds
    start_time = end_time - (days * 24 * 60 * 60 * 1000)  # N days ago in milliseconds

    # Fetch OHLCV data for both symbols
    try:
        data1 = fetch_all_ohlcv(symbol1, timeframe, start_time, end_time)
        data2 = fetch_all_ohlcv(symbol2, timeframe, start_time, end_time)
    except Exception as e:
        return render_template('index.html', error=f"Error fetching data: {str(e)}")

    # Check if data is empty
    if not data1 or not data2:
        return render_template('index.html', error="No data returned from GATE.io. Please check the symbols and timeframe.")

    # Convert OHLCV data to DataFrames
    df1 = pd.DataFrame(data1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df2 = pd.DataFrame(data2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Convert timestamp to datetime
    df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms')
    df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')

    # Set the timestamp as the index
    df1.set_index('timestamp', inplace=True)
    df2.set_index('timestamp', inplace=True)

    # Merge data on timestamp
    data = pd.merge(df1['close'], df2['close'], left_index=True, right_index=True, suffixes=('_1', '_2'))

    # Drop rows with missing values
    data_cleaned = data.dropna()

    # Test for cointegration
    score, pvalue, _ = coint(data_cleaned.iloc[:, 0], data_cleaned.iloc[:, 1])

    # Calculate the spread and Z-score
    spread = data_cleaned.iloc[:, 0] - data_cleaned.iloc[:, 1]
    mean_spread = spread.mean()
    std_spread = spread.std()
    zscore = (spread - mean_spread) / std_spread

    # Generate the Z-Score plot
    plot_data = generate_zscore_plot(symbol1, symbol2, zscore, days)

    # Return data to render template with plot and values
    response = {
        'symbol1': symbol1,
        'symbol2': symbol2,
        'cointegration_pvalue': pvalue,
        'zscore': zscore.tail().to_dict(),
        'signals': {
            'buy': zscore[zscore < -2].index.strftime('%Y-%m-%d %H:%M').tolist(),
            'sell': zscore[zscore > 2].index.strftime('%Y-%m-%d %H:%M').tolist(),
        },
        'plot': plot_data,  # Include the Plotly chart
        'timeframe': timeframe,
        'days': days
    }

    return render_template('index.html', response=response)

# Endpoint to update the plot
@app.route('/update_plot', methods=['GET'])
def update_plot():
    symbol1 = request.args.get('symbol1', 'DOGE/USDT')
    symbol2 = request.args.get('symbol2', 'XLM/USDT')
    timeframe = request.args.get('timeframe', '5m')
    days = int(request.args.get('days', 30))

    # Same data fetching and processing steps as in the home route
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    data1 = fetch_all_ohlcv(symbol1, timeframe, start_time, end_time)
    data2 = fetch_all_ohlcv(symbol2, timeframe, start_time, end_time)

    df1 = pd.DataFrame(data1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df2 = pd.DataFrame(data2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms')
    df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')

    df1.set_index('timestamp', inplace=True)
    df2.set_index('timestamp', inplace=True)

    data = pd.merge(df1['close'], df2['close'], left_index=True, right_index=True, suffixes=('_1', '_2'))
    data_cleaned = data.dropna()

    score, pvalue, _ = coint(data_cleaned.iloc[:, 0], data_cleaned.iloc[:, 1])

    spread = data_cleaned.iloc[:, 0] - data_cleaned.iloc[:, 1]
    mean_spread = spread.mean()
    std_spread = spread.std()
    zscore = (spread - mean_spread) / std_spread

    # Generate updated Z-Score plot
    plot_data = generate_zscore_plot(symbol1, symbol2, zscore, days)

    # Inside the update_plot function
    zscore_dict = zscore.tail().to_dict()
    zscore_dict = {str(key): value for key, value in zscore_dict.items()}  # Convert Timestamp keys to string

    return jsonify({
        'zscore': zscore_dict,
        'plot': plot_data,
    })

def fetch_all_ohlcv(symbol, timeframe, start_time, end_time):
    all_data = []
    while start_time < end_time:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_time, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            start_time = ohlcv[-1][0] + 1
        except ccxt.NetworkError as e:
            time.sleep(5)
        except ccxt.ExchangeError as e:
            break
    return all_data


import requests
from apscheduler.schedulers.background import BackgroundScheduler

# Configuration du bot Telegram
TELEGRAM_BOT_TOKEN = "7922194313:AAG_JukaF96s7rFTt5GDHKExtYPN-ZnW-is"
TELEGRAM_CHAT_ID = "1173292031"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

def update_zscore():
    symbol1 = "DOGE/USDT"
    symbol2 = "XLM/USDT"
    timeframe = "1m"
    days = 5

    try:
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)

        data1 = fetch_all_ohlcv(symbol1, timeframe, start_time, end_time)
        data2 = fetch_all_ohlcv(symbol2, timeframe, start_time, end_time)

        df1 = pd.DataFrame(data1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df2 = pd.DataFrame(data2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms')
        df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')

        df1.set_index('timestamp', inplace=True)
        df2.set_index('timestamp', inplace=True)

        data = pd.merge(df1['close'], df2['close'], left_index=True, right_index=True, suffixes=('_1', '_2'))
        data_cleaned = data.dropna()

        spread = data_cleaned.iloc[:, 0] - data_cleaned.iloc[:, 1]
        mean_spread = spread.mean()
        std_spread = spread.std()
        zscore = (spread - mean_spread) / std_spread

        latest_zscore = zscore.iloc[-1]  # Dernier Z-score

        print(f"🔄 Mise à jour du Z-score: {latest_zscore}")
        send_telegram_message(f"La valeur du z score est de {latest_zscore:.2f}.")
        # Vérifier si le Z-score dépasse 1.8 ou -1.8
        if latest_zscore > 1.8:
            send_telegram_message(f"⚠️ ALERTE: Le Z-score ({symbol1} - {symbol2}) est de {latest_zscore:.2f} (>1.8). Envisagez de vendre {symbol1}.")
        elif latest_zscore < -1.8:
            send_telegram_message(f"⚠️ ALERTE: Le Z-score ({symbol1} - {symbol2}) est de {latest_zscore:.2f} (<-1.8). Envisagez d'acheter {symbol1}.")

    except Exception as e:
        print(f"Erreur lors de la mise à jour du Z-score: {e}")


scheduler = BackgroundScheduler()
scheduler.add_job(update_zscore, 'interval', minutes=1)
scheduler.start()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Utilise le port de Render ou 10000 par défaut
    app.run(host="0.0.0.0", port=port)
    """

from flask import Flask, render_template, request, jsonify
import ccxt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import time
import matplotlib.pyplot as plt
import base64
import io
import plotly.graph_objs as go
import os
import requests
from apscheduler.schedulers.background import BackgroundScheduler

# Set Matplotlib to use a non-GUI backend
plt.switch_backend('Agg')

app = Flask(__name__)

# Initialize GATE.io futures exchange
exchange = ccxt.gateio({
    'options': {
        'defaultType': 'future',  # Use the futures API
    }
})

# Function to generate the Z-Score Plotly chart with zoom and pan features
def generate_zscore_plot(symbol1, symbol2, zscore, days):
    # Create the Z-Score plot using Plotly
    trace = go.Scatter(
        x=zscore.index,
        y=zscore,
        mode='lines',
        name='Z-Score',
        line=dict(color='blue', width=1.5),
        hoverinfo='x+y',  # Show the x and y values when hovering
    )

    # Create threshold lines for the plot
    upper_threshold = go.Scatter(
        x=zscore.index,
        y=[2] * len(zscore),
        mode='lines',
        name='Upper Threshold (2)',
        line=dict(color='red', dash='dash'),
        showlegend=False,  # Do not display in legend
    )

    lower_threshold = go.Scatter(
        x=zscore.index,
        y=[-2] * len(zscore),
        mode='lines',
        name='Lower Threshold (-2)',
        line=dict(color='green', dash='dash'),
        showlegend=False,  # Do not display in legend
    )

    mean_line = go.Scatter(
        x=zscore.index,
        y=[0] * len(zscore),
        mode='lines',
        name='Mean (0)',
        line=dict(color='black', dash='dash'),
        showlegend=False,  # Do not display in legend
    )

    # Combine traces into a figure
    fig = go.Figure(data=[trace, upper_threshold, lower_threshold, mean_line])

    # Update layout with title, axis labels, etc.
    fig.update_layout(
        title=f"Z-Score of the Spread ({symbol1} - {symbol2}) - Last {days} Days",
        xaxis_title="Date and Time",
        yaxis_title="Z-Score",
        xaxis=dict(
            tickformat="%Y-%m-%d %H:%M",
            showgrid=True,
        ),
        yaxis=dict(
            showgrid=True,
        ),
        hovermode="closest",  # Tooltip will show when hovering over the points
        dragmode='zoom',  # Enable zoom and pan mode (similar to TradingView)
        template="plotly_white",  # White theme for the plot
        autosize=True,  # Automatically resize the chart
        margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins for better readability
        height=800,  # Increase the height for a larger chart
        width=1200,  # Increase the width for a larger chart
    )

    # Return the plot HTML div for embedding in the template
    return fig.to_html(full_html=False)

@app.route('/', methods=['GET', 'POST'])
def home():
    # Default values
    symbol1 = request.args.get('symbol1', 'DOGE/USDT')
    symbol2 = request.args.get('symbol2', 'XLM/USDT')
    timeframe = request.args.get('timeframe', '1m')  # Get timeframe from query parameters
    days = int(request.args.get('days', 5))  # Get days from query parameters

    # Calculate the start and end times
    end_time = int(time.time() * 1000)  # Current time in milliseconds
    start_time = end_time - (days * 24 * 60 * 60 * 1000)  # N days ago in milliseconds

    # Fetch OHLCV data for both symbols
    try:
        data1 = fetch_all_ohlcv(symbol1, timeframe, start_time, end_time)
        data2 = fetch_all_ohlcv(symbol2, timeframe, start_time, end_time)
    except Exception as e:
        return render_template('index.html', error=f"Error fetching data: {str(e)}")

    # Check if data is empty
    if not data1 or not data2:
        return render_template('index.html', error="No data returned from GATE.io. Please check the symbols and timeframe.")

    # Convert OHLCV data to DataFrames
    df1 = pd.DataFrame(data1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df2 = pd.DataFrame(data2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Convert timestamp to datetime
    df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms')
    df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')

    # Set the timestamp as the index
    df1.set_index('timestamp', inplace=True)
    df2.set_index('timestamp', inplace=True)

    # Merge data on timestamp
    data = pd.merge(df1['close'], df2['close'], left_index=True, right_index=True, suffixes=('_1', '_2'))

    # Drop rows with missing values
    data_cleaned = data.dropna()

    # Test for cointegration
    score, pvalue, _ = coint(data_cleaned.iloc[:, 0], data_cleaned.iloc[:, 1])

    # Calculate the spread and Z-score
    spread = data_cleaned.iloc[:, 0] - data_cleaned.iloc[:, 1]
    mean_spread = spread.mean()
    std_spread = spread.std()
    zscore = (spread - mean_spread) / std_spread

    # Generate the Z-Score plot
    plot_data = generate_zscore_plot(symbol1, symbol2, zscore, days)

    # Return data to render template with plot and values
    response = {
        'symbol1': symbol1,
        'symbol2': symbol2,
        'cointegration_pvalue': pvalue,
        'zscore': zscore.tail().to_dict(),
        'signals': {
            'buy': zscore[zscore < -2].index.strftime('%Y-%m-%d %H:%M').tolist(),
            'sell': zscore[zscore > 2].index.strftime('%Y-%m-%d %H:%M').tolist(),
        },
        'plot': plot_data,  # Include the Plotly chart
        'timeframe': timeframe,
        'days': days
    }

    return render_template('index.html', response=response)

# Endpoint to update the plot
@app.route('/update_plot', methods=['GET'])
def update_plot():
    symbol1 = request.args.get('symbol1', 'DOGE/USDT')
    symbol2 = request.args.get('symbol2', 'XLM/USDT')
    timeframe = request.args.get('timeframe', '5m')
    days = int(request.args.get('days', 30))

    # Same data fetching and processing steps as in the home route
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    data1 = fetch_all_ohlcv(symbol1, timeframe, start_time, end_time)
    data2 = fetch_all_ohlcv(symbol2, timeframe, start_time, end_time)

    df1 = pd.DataFrame(data1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df2 = pd.DataFrame(data2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms')
    df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')

    df1.set_index('timestamp', inplace=True)
    df2.set_index('timestamp', inplace=True)

    data = pd.merge(df1['close'], df2['close'], left_index=True, right_index=True, suffixes=('_1', '_2'))
    data_cleaned = data.dropna()

    score, pvalue, _ = coint(data_cleaned.iloc[:, 0], data_cleaned.iloc[:, 1])

    spread = data_cleaned.iloc[:, 0] - data_cleaned.iloc[:, 1]
    mean_spread = spread.mean()
    std_spread = spread.std()
    zscore = (spread - mean_spread) / std_spread

    # Generate updated Z-Score plot
    plot_data = generate_zscore_plot(symbol1, symbol2, zscore, days)

    # Inside the update_plot function
    zscore_dict = zscore.tail().to_dict()
    zscore_dict = {str(key): value for key, value in zscore_dict.items()}  # Convert Timestamp keys to string

    return jsonify({
        'zscore': zscore_dict,
        'plot': plot_data,
    })

def fetch_all_ohlcv(symbol, timeframe, start_time, end_time):
    all_data = []
    while start_time < end_time:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_time, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            start_time = ohlcv[-1][0] + 1
        except ccxt.NetworkError as e:
            time.sleep(5)
        except ccxt.ExchangeError as e:
            break
    return all_data

# Configuration du bot Telegram
TELEGRAM_BOT_TOKEN = "7922194313:AAG_JukaF96s7rFTt5GDHKExtYPN-ZnW-is"
TELEGRAM_CHAT_ID = "1173292031"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

def update_zscore():
    symbol1 = "DOGE/USDT"
    symbol2 = "XLM/USDT"
    timeframe = "1m"
    days = 5

    try:
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)

        data1 = fetch_all_ohlcv(symbol1, timeframe, start_time, end_time)
        data2 = fetch_all_ohlcv(symbol2, timeframe, start_time, end_time)

        df1 = pd.DataFrame(data1, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df2 = pd.DataFrame(data2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms')
        df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')

        df1.set_index('timestamp', inplace=True)
        df2.set_index('timestamp', inplace=True)

        data = pd.merge(df1['close'], df2['close'], left_index=True, right_index=True, suffixes=('_1', '_2'))
        data_cleaned = data.dropna()

        spread = data_cleaned.iloc[:, 0] - data_cleaned.iloc[:, 1]
        mean_spread = spread.mean()
        std_spread = spread.std()
        zscore = (spread - mean_spread) / std_spread

        latest_zscore = zscore.iloc[-1]  # Dernier Z-score

        print(f"🔄 Mise à jour du Z-score: {latest_zscore}")
        send_telegram_message(f"La valeur du z score est de {latest_zscore:.2f}.")
        # Vérifier si le Z-score dépasse 1.8 ou -1.8
        if latest_zscore > 1.8:
            send_telegram_message(f"⚠️ ALERTE: Le Z-score ({symbol1} - {symbol2}) est de {latest_zscore:.2f} (>1.8). Envisagez de vendre {symbol1}.")
        elif latest_zscore < -1.8:
            send_telegram_message(f"⚠️ ALERTE: Le Z-score ({symbol1} - {symbol2}) est de {latest_zscore:.2f} (<-1.8). Envisagez d'acheter {symbol1}.")

    except Exception as e:
        print(f"Erreur lors de la mise à jour du Z-score: {e}")

# Endpoint pour le keep-alive
@app.route('/keep-alive', methods=['GET'])
def keep_alive():
    return jsonify({"status": "ok"}), 200

# Fonction pour envoyer une requête keep-alive
def send_keep_alive():
    try:
        response = requests.get("https://votre-service.onrender.com/keep-alive")
        print(f"Keep-alive request sent. Status code: {response.status_code}")
    except Exception as e:
        print(f"Erreur lors de l'envoi de la requête keep-alive: {e}")

# Configurer le scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(update_zscore, 'interval', minutes=1)  # Mise à jour du Z-score toutes les 1 minute
scheduler.add_job(send_keep_alive, 'interval', minutes=10)  # Envoyer une requête keep-alive toutes les 10 minutes
scheduler.start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Utilise le port de Render ou 10000 par défaut
    app.run(host="0.0.0.0", port=port)
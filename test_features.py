
import pandas as pd 
import numpy as np 
from pandas import DataFrame
import logging
from typing import Dict, List
from tqdm import tqdm
from joblib import Parallel, delayed
# from tqdm_joblib import tqdm_joblib
import traceback
import itertools


import cartesiuslib.feature_extraction as clib_features


""" 
Implemetação das novas features Rafare + Felipe 

PARTE 1: Funções específicas de cada feature
PARTE 2: Funções processamento geral 
PARTE 3: Features Guilherme 
PARTE 4: Features janeladas Felipe

código feito por: Victor Siqueira

"""


#########################################################
# 1) Funções específicas de cada feature
#########################################################

# 1 - KAUFMAN'S ADAPTATIVE MOVING AVERAGE (KAMA)
def kaufmans_adaptive_moving_average(close: pd.Series, window: int, fast: int = 2, slow: int = 30) -> pd.Series:
    er = abs(close - close.shift(window)) / (close.diff().abs().rolling(window=window).sum())
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    kama = pd.Series(index=close.index, dtype=float)
    kama.iloc[window-1] = close.iloc[window-1]
    for i in range(window, len(close)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        
    return kama


#  2 - STOCHASTIC RSI (COMBINAÇÃO DO RSI COM O OSCILADOR ESTOCÁSTICO)
def stochastic_rsi(close: pd.Series, period: int, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    rsi = relative_strength_index(close, period)
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return pd.DataFrame({'StochRSI_K': k, 'StochRSI_D': d})

#  3 - ehlers_fisher_transform 
def ehlers_fisher_transform(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    hl2 = (high + low) / 2
    value = (2 * ((hl2 - hl2.rolling(window=period).min()) / (hl2.rolling(window=period).max() - hl2.rolling(window=period).min())) - 1)
    smooth = value.ewm(span=5).mean()
    return 0.5 * np.log((1 + smooth) / (1 - smooth))


# 4 - FORECAST OSCILLATOR
def forecast_oscillator(close: pd.Series, period: int) -> pd.Series:
    linear_reg = close.rolling(window=period).apply(lambda x: np.polyfit(range(period), x, 1)[0])
    return (close - (close.rolling(window=period).mean() + linear_reg * (period-1)/2)) / close.rolling(window=period).std()


# 5 - PRETTY GOOD OSCILLATOR    
def pretty_good_oscillator(close: pd.Series, period: int) -> pd.Series:
    return (close - close.rolling(window=period).min()) / (close.rolling(window=period).max() - close.rolling(window=period).min())

#  ( Já possui na ctlib, porém utilizo para calcular o stocastich RSI )
# 6 - RSI 
def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 6 - VORTEX INDICATOR  
def vortex_indicator(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
    vm_plus = np.abs(high - low.shift(1))
    vm_minus = np.abs(low - high.shift(1))
    
    vi_plus = vm_plus.rolling(window=window).sum() / tr.rolling(window=window).sum()
    vi_minus = vm_minus.rolling(window=window).sum() / tr.rolling(window=window).sum()
    
    return pd.DataFrame({'VI_plus': vi_plus, 'VI_minus': vi_minus})

# 7 - ELDER'S THERMOMETER
def elders_thermometer(close: pd.Series, period: int) -> pd.Series:
    return (close - close.rolling(window=period).min()) / (close.rolling(window=period).max() - close.rolling(window=period).min()) * 100


# 8 - RATE OF CHANGE
def rate_of_change(close: pd.Series, window: int) -> pd.Series:
    return (close - close.shift(window)) / close.shift(window) * 100


# 9 - ZERO LAG EXPONENTIAL MOVING AVERAGE (ZLEMA)
def zero_lag_ema(close: pd.Series, period: int) -> pd.Series:
    lag = (period - 1) // 2
    return 2 * close.ewm(span=period).mean() - close.ewm(span=period).mean().shift(lag)

# 10 - ÍNDICE DE VIGOR RELATIVO
def relative_vigor_index(open: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    num = (close - open).rolling(window=period).sum()
    den = (high - low).rolling(window=period).sum()
    return num / den


# 11 - RATE OF CHANGE
def rate_of_change(close: pd.Series, window: int) -> pd.Series:
    return (close - close.shift(window)) / close.shift(window) * 100


# 12 - STOCHASTIC OSCILLATOR
def stochastic_oscillator(close: pd.Series, high: pd.Series, low: pd.Series, window: int) -> pd.DataFrame:
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=3).mean()
    return pd.DataFrame({
        'Stochastic_K': k,
        'Stochastic_D': d
    })

####################################################################
# 2) funções necessárias processamento geral: 
####################################################################

def validate_features(df): # valida as colunas
    problematic_cols = identify_problematic_columns(df)
    if problematic_cols:
        logging.warning(f"Columns with all NaN or infinite values: {problematic_cols}")
        # You can choose to drop these columns or fill them with a default value
        df = df.drop(columns=problematic_cols)
    return df


def treat_nans(df: pd.DataFrame) -> pd.DataFrame: # função para tratar os valores nan das funções 
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

    
# MAPEAMENTO DAS COLUNAS DE PREÇOS DA PLANILHA (A PLANILHA USA MAX E MIN)
def map_ohlc_columns(df: pd.DataFrame) -> Dict[str, str]:
    ohlc_map = {
        'Open': 'Open',
        'Close': 'Close',
        'High': 'Max',
        'Low': 'Min'
    }
    for expected, actual in ohlc_map.items():
        if actual not in df.columns:
            logging.warning(f"Aviso: Coluna '{actual}' não encontrada. Verificando alternativas...")
            alternatives = [col for col in df.columns if expected.lower() in col.lower()]
            if alternatives:
                ohlc_map[expected] = alternatives[0]
                logging.info(f"Usando '{alternatives[0]}' para '{expected}'")
            else:
                logging.error(f"Erro: Não foi possível encontrar uma coluna para '{expected}'")
                ohlc_map[expected] = None
    
    logging.info("\nMapeamento final das colunas OHLC:")
   #logging.info(str(ohlc_map))
    
    return ohlc_map

def safe_calculate(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            logging.debug(f"Function {func.__name__} returned DataFrame with shape: {result.shape}")
        elif isinstance(result, pd.Series):
            logging.debug(f"Function {func.__name__} returned Series with length: {len(result)}")
        return result
    except Exception as e:
        logging.error(f"Error calculating {func.__name__}: {str(e)}")
        logging.error(f"Arguments: {args}, Keyword arguments: {kwargs}")
        logging.error(f"Error details: {traceback.format_exc()}")
        # Retornar uma Series de NaN com o mesmo índice da série de entrada
        if len(args) > 0 and isinstance(args[0], pd.Series):
            return pd.Series(np.nan, index=args[0].index)
        else:
            return pd.Series(np.nan)


def identify_problematic_columns(df): # verifica se todos os valores são nan para validar colunas 
    problematic_cols = []
    for col in df.columns:
        if df[col].isna().all() :
            problematic_cols.append(col)
    return problematic_cols
# or np.isinf(df[col]).all()

#  FUNÇÃO DE PREPROCESSAMENTO DOS DADOS 
def pre_process_data(df:pd.DataFrame) -> pd.DataFrame:
    
    try:
        # Load data
        logging.info(f"Shape of input data: {df.shape}")
        
        
        # Validate and clean features
        new_features_df = validate_features(df)
        
        df_final_processed = new_features_df
        # print(f"df_final_processed:{type(df_final_processed)}")
        
        # Handle NaNs
        df_final = treat_nans(df_final_processed)
        logging.info(f"Shape after NaN treatment: {df_final.shape}")
        # period_teste =df_final["period"]
        
        
        df_normalized = df 
    
        
        return df_normalized

    except Exception as e:
        logging.error(f"An error occurred during data processing: {str(e)}")
        raise
    
###############################################################################
            # 3) CÁLCULO DAS FUNÇÕES DAS NOVAS FEATURES: 
###############################################################################




################################################################################
             # CÁLCULOS DO STOCHASTIC OSCILLATOR
###############################################################################


def new_stochastic_oscillator(df:pd.DataFrame,janela=[5])-> DataFrame:
        
        
    df = pre_process_data(df) # preprocessamento dos dados 
    ohlc_map= map_ohlc_columns(df)
    new_features = {}
    new_features = pd.DataFrame(index=df.index)

    stochastic_periods =  janela 
    
    if all(ohlc_map.get(key) is not None for key in ['Close', 'High', 'Low']):
        for period in tqdm(stochastic_periods, desc="Calculating Stochastic Oscillators"):
            stoch_df = safe_calculate(
                stochastic_oscillator,
                df[ohlc_map['Close']],
                df[ohlc_map['High']],
                df[ohlc_map['Low']],
                period
            )
            new_features[f'Stochastic_K_{period}'] = stoch_df['Stochastic_K']
            new_features[f'Stochastic_D_{period}'] = stoch_df['Stochastic_D']
            logging.info(f"Stochastic Oscillator for period {period} created")
                
        return new_features

################################################################################
        # CÁLCULOS DO FORECAST OSCILLATOR
################################################################################


def forecast_oscilator(df:pd.DataFrame,janela=[14])-> DataFrame:
        
        df = pre_process_data(df) # preprocessamento dos dados 
        ohlc_map= map_ohlc_columns(df)
        new_features = {}
        logging.info("Starting pretty_good_oscilator feature calculation...")
        new_features = pd.DataFrame(index=df.index)
        
            
        forecast_oscillator_periods = janela
        if 'Close' in ohlc_map:
            for period in tqdm(forecast_oscillator_periods, desc="Calculating Forecast Oscillator"):
                
                close_prices = df[ohlc_map['Close']]
                
                # Cálculo do Forecast Oscillator
                fo = forecast_oscillator(close_prices, period)
                
                # Cálculo do sinal
                signal = pd.Series(0, index=fo.index)
                signal[fo > 0] = 1  # Sinal de compra
                signal[fo < 0] = -1  # Sinal de venda
                
                # Cálculo da média móvel
                fo_ma = fo.rolling(window=10).mean()
                
                # new_features[f'Forecast_Oscillator_{period}'] = fo
                new_features[f'FO_Signal_{period}'] = signal
                # new_features[f'FO_MA_{period}'] = fo_ma
                
                logging.info(f"Forecast Oscillator with period {period} created successfully")
                    
                
        return new_features

################################################################################
            # CÁLCULOS DO ÍNDICE DIRECIONAL MÉDIO (ADX)
################################################################################

def adx (df:pd.DataFrame,janela=[7])-> DataFrame:

        df = pre_process_data(df) # preprocessamento dos dados 
        ohlc_map= map_ohlc_columns(df)
        
        new_features = {}
        logging.info("Starting pretty_good_oscilator feature calculation...")
        new_features = pd.DataFrame(index=df.index)
        
            
        adx_periods =  janela
        
        if all(col in ohlc_map for col in ['High', 'Low', 'Close']):
            for period in tqdm(adx_periods, desc="Calculating ADX"):
                try:
                    high = df[ohlc_map['High']]
                    low = df[ohlc_map['Low']]
                    close = df[ohlc_map['Close']]
                    
                    # Cálculo do True Range (TR)
                    tr1 = high - low
                    tr2 = abs(high - close.shift(1))
                    tr3 = abs(low - close.shift(1))
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.rolling(window=period).mean()
                    
                    # Cálculo do Directional Movement (DM)
                    up_move = high - high.shift(1)
                    down_move = low.shift(1) - low
                    
                    pos_dm = ((up_move > down_move) & (up_move > 0)) * up_move
                    neg_dm = ((down_move > up_move) & (down_move > 0)) * down_move
                    
                    # Cálculo do Directional Indicator (DI)
                    di_plus = 100 * pos_dm.rolling(window=period).mean() / atr
                    di_minus = 100 * neg_dm.rolling(window=period).mean() / atr
                    
                    # Cálculo do DX e ADX
                    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
                    adx = dx.rolling(window=period).mean()
                    
                    # new_features[f'ADX_{period}'] = adx
                    new_features[f'Plus_DI_{period}'] = di_plus
                    # new_features[f'Minus_DI_{period}'] = di_minus
                    
                    logging.info(f"ADX with period {period} created successfully")
                except Exception as e:
                    logging.error(f"Error calculating ADX for period {period}: {str(e)}")
                    logging.error(f"Error type: {type(e).__name__}")
                    logging.error(f"Error details: {traceback.format_exc()}")
                    new_features[f'ADX_{period}'] = np.nan
                    new_features[f'Plus_DI_{period}'] = np.nan
                    new_features[f'Minus_DI_{period}'] = np.nan
                
        return new_features

################################################################################
                        # CÁLCULOS DO SUPERTREND_5
################################################################################
        
def supertrend(df:pd.DataFrame,janela=[5])-> DataFrame:
        
        df = pre_process_data(df) # preprocessamento dos dados 
        ohlc_map= map_ohlc_columns(df)
        
        new_features = {}
        logging.info("Starting pretty_good_oscilator feature calculation...")
        new_features = pd.DataFrame(index=df.index)
        
            
        supertrend_periods = janela  
        supertrend_multiplier = 3.0  
        
        if all(col in ohlc_map for col in ['Close', 'High', 'Low']):
            for period in tqdm(supertrend_periods, desc="Calculating Supertrend"):
                try:
                    supertrend_values = safe_calculate(
                        supertrend,
                        df[ohlc_map['Close']],
                        df[ohlc_map['High']],
                        df[ohlc_map['Low']],
                        period,
                        supertrend_multiplier
                    )
                    new_features[f'Supertrend_{period}'] = supertrend_values['Supertrend']
                    new_features[f'Supertrend_Trend_{period}'] = supertrend_values['Trend']
                    logging.info(f"Supertrend with period {period} created successfully")
                except Exception as e:
                    logging.error(f"Error calculating Supertrend for period {period}: {str(e)}")
                    logging.error(f"Error type: {type(e).__name__}")
                    logging.error(f"Error details: {traceback.format_exc()}")
                    new_features[f'Supertrend_{period}'] = np.nan
                    new_features[f'Supertrend_Trend_{period}'] = np.nan
        
        return new_features
                
    

################################################################################
                # CÁLCULOS DO PRETTY GOOD OSCILLATOR
################################################################################

def pretty_good_oscilator(df: DataFrame, janela:List=[20]) -> DataFrame:
    
        # janela por default será 20 
    
        df = pre_process_data(df) # preprocessamento dos dados 
        
        new_features = {}
        logging.info("Starting pretty_good_oscilator feature calculation...")
        new_features = pd.DataFrame(index=df.index)
            
        ohlc_map= map_ohlc_columns(df)
        
        pretty_good_oscillator_periods = janela
        
        if 'Close' in ohlc_map:
            for period in tqdm(pretty_good_oscillator_periods, desc="Calculating Pretty Good Oscillator"):
                try:
                    close_prices = df[ohlc_map['Close']]
                    
                    # Cálculo do Pretty Good Oscillator
                    pgo = pretty_good_oscillator(close_prices, period)
                    
                    # Cálculo do sinal
                    signal = pd.Series(0, index=pgo.index)
                    signal[pgo > 0] = 1  # Sinal de compra
                    signal[pgo < 0] = -1  # Sinal de venda
                    
                    # Cálculo da média móvel
                    pgo_ma = pgo.rolling(window=10).mean()
                    
                    new_features[f'Pretty_Good_Oscillator_{period}'] = pgo
                    new_features[f'PGO_Signal_{period}'] = signal
                    new_features[f'PGO_MA_{period}'] = pgo_ma
                    
                    logging.info(f"Pretty Good Oscillator with period {period} created successfully")
                except Exception as e:
                    logging.error(f"Error calculating Pretty Good Oscillator for period {period}: {str(e)}")
                    logging.error(f"Error type: {type(e).__name__}")
                    logging.error(f"Error details: {traceback.format_exc()}")
                    new_features[f'Pretty_Good_Oscillator_{period}'] = np.nan
                    new_features[f'PGO_Signal_{period}'] = np.nan
                    new_features[f'PGO_MA_{period}'] = np.nan
                
            
        return new_features 
    
################################################################################
                # CÁLCULOS DO RELATIVE STRENGTH INDEX (RSI)
################################################################################
        # Definir flags no início do script ou em uma seção de configuração


# mantenho essa função, pois uso para calculo do RSI Stocastic  
def relative_strength_index(df: DataFrame,period)-> DataFrame:
    
    # janela por default será 5 
    
    df = pre_process_data(df) # preprocessamento dos dados 
          
    new_features = {}
    logging.info("Starting pretty_good_oscilator feature calculation...")
    new_features = pd.DataFrame(index=df.index)
    
    period_list=[period]
    rsi_periods = period_list  # Assumindo os mesmos períodos dos outros indicadores
    ohlc_map= map_ohlc_columns(df)
    
    if ohlc_map.get('Close') is not None:
        for period in rsi_periods:
            rsi_values = safe_calculate(
                rsi,
                df[ohlc_map['Close']],
                period
            )
            new_features[f'RSI_{period}'] = rsi_values
                
        
    return new_features

################################################################################
                    # CÁLCULOS DO VORTEX INDICATOR
################################################################################
        
def votex_indicator(df: DataFrame, janela:List=[7])-> DataFrame:
    
    
    # janela por default será 7
    
    df = pre_process_data(df) # preprocessamento dos dados 
    
    ohlc_map= map_ohlc_columns(df)
          
    new_features = {}
    logging.info("Starting pretty_good_oscilator feature calculation...")
    new_features = pd.DataFrame(index=df.index)

    vortex_periods = janela  
    
    if all(col in ohlc_map for col in ['Close', 'High', 'Low']):
        for period in tqdm(vortex_periods, desc="Calculating Vortex Indicator"):
            try:
                vortex_values = safe_calculate(
                    vortex_indicator,
                    df[ohlc_map['High']],
                    df[ohlc_map['Low']],
                    df[ohlc_map['Close']],
                    period
                )
                new_features[f'VI_plus_{period}'] = vortex_values['VI_plus']
                new_features[f'VI_minus_{period}'] = vortex_values['VI_minus']
                logging.info(f"Vortex Indicator with period {period} created successfully")
            except Exception as e:
                print(f" Error na função votex_indicator: {e}")
            
        
    return new_features

################################################################################
        # CÁLCULOS DO STOCHASTIC RELATIVE STRENGTH INDEX (Stochastic RSI)
################################################################################

        
def stochastic_rsi_new(df: DataFrame, janela:List=[5])-> DataFrame:
    
    # janela por default será 5 
    
    df = pre_process_data(df) # preprocessamento dos dados 
    
    ohlc_map= map_ohlc_columns(df)
          
    new_features = {}
    new_features = pd.DataFrame(index=df.index)
    
    stoch_rsi_periods = janela 
    
    for period in stoch_rsi_periods:
        if ohlc_map.get('Close') is not None:
            # for period in tqdm(stoch_rsi_periods, desc="Calculating Stochastic RSI"):
            for period in stoch_rsi_periods:
                stoch_rsi_result = safe_calculate(
                    stochastic_rsi,
                    df[['Close']],
                    period
                )
                
                # stoch_rsi_result = stochastic_rsi(df[['Close']],5)
                # print(len(stoch_rsi_result.isna()),len(stoch_rsi_result))
                new_features[f'StochRSI_K_{period}'] = stoch_rsi_result['StochRSI_K']
                new_features[f'StochRSI_D_{period}'] = stoch_rsi_result['StochRSI_D']
                # logging.info(f"StochRSI_K_{period} and StochRSI_D_{period} columns created")
             
        
            
    return new_features

################################################################################
                 # CÁLCULOS DO ELDERS THERMOMETER
################################################################################
def elsers_thermometer(df: DataFrame, janela:List=[50])-> DataFrame:
        
        
        
        new_features = {}
        logging.info("Starting pretty_good_oscilator feature calculation...")
        new_features = pd.DataFrame(index=df.index)
    
        # janela por default será 5 
        df = pre_process_data(df) # preprocessamento dos dados 
        ohlc_map= map_ohlc_columns(df) 
        elders_thermometer_periods =  janela
        
        if 'Close' in ohlc_map:
            for period in tqdm(elders_thermometer_periods, desc="Calculating Elders Thermometer"):
                try:
                    close_prices = df[ohlc_map['Close']]
                    
                    # Cálculo do Elders Thermometer
                    et = elders_thermometer(close_prices, period)
                    
                    # Cálculo do sinal
                    signal = pd.Series(0, index=et.index)
                    signal[et > 0] = 1  # Sinal de compra
                    signal[et < 0] = -1  # Sinal de venda
                    
                    # Cálculo da média móvel
                    et_ma = et.rolling(window=10).mean()
                    
                    new_features[f'Elders_Thermometer_{period}'] = et
                    new_features[f'ET_Signal_{period}'] = signal
                    new_features[f'ET_MA_{period}'] = et_ma
                    
                    logging.info(f"Elders Thermometer with period {period} created successfully")
                except Exception as e:
                    
                    print(f" Error na função elsers_thermometer{e}")               
            
        return new_features
        
################################################################################
                        # CÁLCULOS DO ELDER RAY INDEX
################################################################################
        
def elder_ray_index(df: DataFrame, janela:List=[13])-> DataFrame:

    
    # pre_processamento dos dados
    new_features = {}
    new_features = pd.DataFrame(index=df.index)
    df = pre_process_data(df) # preprocessamento dos dados 
    ohlc_map= map_ohlc_columns(df)
        
    elder_ray_periods =  janela 
    
    if all(col in ohlc_map for col in ['Close', 'High', 'Low']):
        for period in tqdm(elder_ray_periods, desc="Calculating Elder Ray Index"):
            try:
                # Calculando a média móvel exponencial (EMA)
                ema_ = df[ohlc_map['Close']].ewm(span=period, adjust=False).mean()
                
                # Calculando Bull Power e Bear Power
                bull_power = df[ohlc_map['High']] - ema_
                bear_power = df[ohlc_map['Low']] - ema_
                
                new_features[f'ERI_bull_power_{period}'] = bull_power
                new_features[f'ERI_bear_power_{period}'] = bear_power
                logging.info(f"Elder Ray Index with period {period} created successfully")
            except Exception as e:
                logging.error(f"Error calculating Elder Ray Index for period {period}: {str(e)}")
                    
            
    return new_features


################################################################################
        # CÁLCULOS DO FISHER TRANSFORM
################################################################################

def fisher_transform(df:pd.DataFrame, janela=[7]):
        
        new_features = {}
        new_features = pd.DataFrame(index=df.index)
        # janela por default será 7

        df = pre_process_data(df) # preprocessamento dos dados
        ohlc_map= map_ohlc_columns(df)
        fisher_transform_periods = janela
        
        if 'High' in ohlc_map and 'Low' in ohlc_map:
            for period in tqdm(fisher_transform_periods, desc="Calculating Fisher Transform"):
                try:
                    fisher_values = safe_calculate(
                        fisher_transform,
                        df[ohlc_map['High']],
                        df[ohlc_map['Low']],
                        period
                    )
                    new_features[f'Fisher_Transform_{period}'] = fisher_values
                    logging.info(f"Fisher Transform with period {period} created")
                except Exception as e:
                    logging.error(f"Error calculating Fisher Transform for period {period}: {str(e)}")
                
        return new_features


################################################################################
        # CÁLCULOS DO EHLERS FISHER TRANSFORM  OBS: ESSA FUNÇAÕ É DIFERENTE DA ANTERIOR
################################################################################

def Ehlers_Fisher(df: DataFrame, janela:List=[30])-> DataFrame: # deixar apenas o Ehlers_Fisher_Transform_30

    new_features = {}
    new_features = pd.DataFrame(index=df.index)
    df = pre_process_data(df) # preprocessamento dos dados 
    ohlc_map= map_ohlc_columns(df)
    
    ehlers_fisher_transform_periods = janela
    
    if all(key in ohlc_map for key in ['High', 'Low']):
        for period in tqdm(ehlers_fisher_transform_periods, desc="Calculating Ehlers Fisher Transform"):
            try:
                high_prices = df[ohlc_map['High']]
                low_prices = df[ohlc_map['Low']]
                
                # Cálculo do Ehlers Fisher Transform
                eft = ehlers_fisher_transform(high_prices, low_prices, period)
                
                # Cálculo do sinal
                signal = pd.Series(0, index=eft.index)
                signal[eft > eft.shift(1)] = 1  # Sinal de compra
                signal[eft < eft.shift(1)] = -1  # Sinal de venda
                
                # Cálculo da média móvel
                # eft_ma = eft.rolling(window=10).mean()
                
                new_features[f'Ehlers_Fisher_Transform_{period}'] = eft
                # new_features[f'EFT_Signal_{period}'] = signal
                # new_features[f'EFT_MA_{period}'] = eft_ma
                
                logging.info(f"Ehlers Fisher Transform with period {period} created successfully")
            except Exception as e:
                print(f"Erro na função Ehlers_Fisher: {e} ")
                new_features[f'Ehlers_Fisher_Transform_{period}'] = np.nan
                new_features[f'EFT_Signal_{period}'] = np.nan
                new_features[f'EFT_MA_{period}'] = np.nan
                                    
        
    return new_features

######################################################################################
        # CÁLCULOS DAS ZERO LAG EXPONENTIAL MOVING AVERAGES (ZLEMAs): # analisar depois 
######################################################################################

def zlemas(df: DataFrame)-> DataFrame:
     
        new_features = {}
        new_features = pd.DataFrame(index=df.index)


        df = pre_process_data(df) # preprocessamento dos dados 

        ohlc_map= map_ohlc_columns(df)
        # Definir flags no início do script ou em uma seção de configuração
        
        # Definição dos períodos para ZLEMA
        zlema_periods = [3, 5, 8, 12, 15, 20, 22, 25, 30]
        
        # Cálculo das ZLEMAs
        for period in tqdm(zlema_periods, desc="Calculating ZLEMAs"):
            zlema_values = safe_calculate(zero_lag_ema, df[ohlc_map['Close']], period)
            new_features[f'ZLEMA_{period}'] = zlema_values
        
        # Cálculo da diferença entre o preço de fechamento e a ZLEMA
        for period in tqdm(zlema_periods, desc="Calculating Close minus ZLEMA"):
            zlema_value = new_features[f'ZLEMA_{period}']
            new_features[f'Close_minus_ZLEMA_{period}'] = (df[ohlc_map['Close']] - zlema_value)/df[ohlc_map['Close']]
        
        # Definição dos períodos para os cruzamentos
        crossings = {
            'short': [8],
            'medium': [20],
            'long': [25]
        }
    
        close = df[ohlc_map['Close']]
    
        # Calcular todas as combinações possíveis
        
        combinations = list(itertools.product(crossings['short'], crossings['medium'], crossings['long']))
    
        for short, medium, long in tqdm(combinations, desc="Calculating ZLEMA Crossings"):
            # Verificar se todas as ZLEMAs necessárias existem
            if all(f'ZLEMA_{period}' in new_features.columns for period in [short, medium, long]):
                # Usar as ZLEMAs existentes
                short_zlema = new_features[f'ZLEMA_{short}']
                medium_zlema = new_features[f'ZLEMA_{medium}']
                long_zlema = new_features[f'ZLEMA_{long}']
                
                # Cálculo do valor do cruzamento
                column_name = f'ZLEMA_Crossing_{short}_{medium}_{long}'
                new_features[column_name] = ((long_zlema - medium_zlema) + (short_zlema - medium_zlema)) / close
            else:
                print(f"Skipping crossing calculation for {short}, {medium}, {long} due to missing ZLEMA")
        
        return new_features
    
    
    
 ################################################################################
        # CÁLCULOS DO RATE OF CHANGE (ROC)
################################################################################

def roc(df: DataFrame,janela=[5])-> DataFrame:
    
        new_features = {}
        new_features = pd.DataFrame(index=df.index)

        # janela por default será 5 

        df = pre_process_data(df) # preprocessamento dos dados 

        ohlc_map= map_ohlc_columns(df)

        roc_periods = janela 
        
        if ohlc_map.get('Close') is not None:
            for period in tqdm(roc_periods, desc="Calculating Rate of Change"):
                roc_values = safe_calculate(
                    rate_of_change,
                    df[ohlc_map['Close']],
                    period
                )
                new_features[f'ROC_{period}'] = roc_values
                logging.info(f"Rate of Change for period {period} created")
                
        return new_features
    
            
            
#####################################################
        # CÁLCULOS DAS MÉDIAS MÓVEIS SIMPLES (SMAs):   
######################################################

def smas(df: DataFrame, janela:List=[5,10,20,30,50])-> DataFrame: 

           
        new_features = {}
        new_features = pd.DataFrame(index=df.index)
        df = pre_process_data(df) # preprocessamento dos dados 
        ohlc_map= map_ohlc_columns(df)
        
        sma_periods = janela
        
        for period in tqdm(sma_periods, desc="Calculating SMAs"):
            sma_value = safe_calculate(sma, df[ohlc_map['Close']], period)
            new_features[f'SMA_{period}'] = sma_value
        
        # CALCULA A DIFERENÇA ENTRE O PREÇO DE FECHAMENTO E A SMA
        for period in tqdm(sma_periods, desc="Calculating Close minus SMA"):
            sma_value = new_features[f'SMA_{period}']
            new_features[f'Close_minus_SMA_{period}'] = (df[ohlc_map['Close']] - sma_value)//df[ohlc_map['Close']]
        
        # Definição dos períodos para os cruzamentos
        crossings = {
            'short': [5],
            'medium': [20],
            'long': [30]
        }
    
        close = df[ohlc_map['Close']]
    
        # Calcular todas as combinações possíveis
        combinations = list(itertools.product(crossings['short'], crossings['medium'], crossings['long']))
    
        for short, medium, long in tqdm(combinations, desc="Calculating SMA Crossings"):
            
            # Usar as SMAs existentes
            short_sma = new_features[f'SMA_{short}']
            medium_sma = new_features[f'SMA_{medium}']
            long_sma = new_features[f'SMA_{long}']
            
            # Cálculo do valor do cruzamento
            column_name = f'SMA_Crossing_{short}_{medium}_{long}'
            new_features[column_name] = ((long_sma - medium_sma) + (short_sma - medium_sma)) / close
              
        
        return new_features
    


################################################################################
        # CÁLCULOS DO EHLERS INSTANTANEOUS TRENDLINE
################################################################################
        
def ehlers_inst_trendline(df: pd.DataFrame, alpha=[0.05,0.15]):
    
    
    new_features = {}
    new_features = pd.DataFrame(index=df.index)
    df = pre_process_data(df) # preprocessamento dos dados 
    ohlc_map= map_ohlc_columns(df)
    
        
    ehlers_instantaneous_trendline_alphas = alpha  
    
    if 'Close' in ohlc_map:
        for alpha in tqdm(ehlers_instantaneous_trendline_alphas, desc="Calculating Ehlers Instantaneous Trendline"):
            
            close = df[ohlc_map['Close']]
            
            # Cálculo do Ehlers Instantaneous Trendline
            it = pd.Series(index=close.index, dtype=float)
            it.iloc[0] = close.iloc[0]
            it.iloc[1] = close.iloc[1]
            
            for i in range(2, len(close)):
                it.iloc[i] = (alpha - alpha**2 / 4) * close.iloc[i] + 0.5 * alpha**2 * close.iloc[i-1] - (alpha - 0.75 * alpha**2) * close.iloc[i-2] + 2 * (1 - alpha) * it.iloc[i-1] - (1 - alpha)**2 * it.iloc[i-2]
            
            # Cálculo do sinal
            signal = pd.Series(0, index=it.index)
            signal[it > it.shift(1)] = 1  # Sinal de compra
            signal[it < it.shift(1)] = -1  # Sinal de venda
            
            # Cálculo da média móvel
            it_ma = it.rolling(window=10).mean()
            
            # new_features[f'Ehlers_Instantaneous_Trendline_{alpha}'] = it
            new_features[f'EIT_Signal_{alpha}'] = signal
            # new_features[f'EIT_MA_{alpha}'] = it_ma
            
            logging.info(f"Ehlers Instantaneous Trendline with alpha {alpha} created successfully")
    
            
    return new_features





##############################################################
    # CÁLCULOS DAS KAUFMAN'S ADAPTATIVE MOVING AVERAGE (KAMAs):
##############################################################


def kamas (df: pd.DataFrame, janela=[8]) :
    
        new_features = {}
        new_features = pd.DataFrame(index=df.index)
        df = pre_process_data(df) # preprocessamento dos dados 
        ohlc_map= map_ohlc_columns(df)
           
        # Definição dos períodos para KAMA, começando de 5
        kama_periods = janela
        
        # Cálculo das KAMAs
        for period in tqdm(kama_periods, desc="Calculating KAMAs"):
            kama_value = safe_calculate(kaufmans_adaptive_moving_average, df[ohlc_map['Close']], period)
            new_features[f'KAMA_{period}'] = kama_value
        
        # Cálculo da diferença entre o preço de fechamento e a KAMA
        for period in tqdm(kama_periods, desc="Calculating Close minus KAMA"):
            kama_value = new_features[f'KAMA_{period}']
            new_features[f'Close_minus_KAMA_{period}'] = (df[ohlc_map['Close']] - kama_value)/df[ohlc_map['Close']]
                
        return new_features
            
    



################################################################################
        # CÁLCULOS DO TRIX (Triple Exponential Average)
################################################################################

def trix (df: pd.DataFrame, janela=[5]):
        
        new_features = {}
        new_features = pd.DataFrame(index=df.index)

        # janela por default será 5 

        df = pre_process_data(df) # preprocessamento dos dados
        ohlc_map= map_ohlc_columns(df) 
        trix_periods = janela
        
        if 'Close' in ohlc_map:
            for period in tqdm(trix_periods, desc="Calculating TRIX"):
                try:
                    close_prices = df[ohlc_map['Close']]
                    
                    # Cálculo do TRIX
                    trix_values = trix(close_prices, period)
                    
                    # Cálculo do sinal
                    signal = pd.Series(0, index=trix_values.index)
                    signal[trix_values > 0] = 1  # Sinal de compra
                    signal[trix_values < 0] = -1  # Sinal de venda
                    
                    # Cálculo da média móvel
                    # trix_ma = trix_values.rolling(window=10).mean()
                    
                    # new_features[f'TRIX_{period}'] = trix_values
                    new_features[f'TRIX_Signal_{period}'] = signal
                    # new_features[f'TRIX_MA_{period}'] = trix_ma
                    
                    logging.info(f"TRIX with period {period} created successfully")
                except Exception as e:
                    logging.error(f"Error calculating TRIX for period {period}: {str(e)}")
                    
        return new_features
    
    

################################################################################
        # CÁLCULOS DO RELATIVE VIGOR INDEX
################################################################################

def vigor_index (df:pd.DataFrame, janela=[35])-> DataFrame:
    
    
        new_features = {}
        new_features = pd.DataFrame(index=df.index)
        df = pre_process_data(df) # preprocessamento dos dados 
        ohlc_map= map_ohlc_columns(df)
            
        rvi_periods = janela
        if all(key in ohlc_map for key in ['Open', 'Close', 'High', 'Low']):
            for period in tqdm(rvi_periods, desc="Calculating Relative Vigor Index"):
                try:
                    open_prices = df[ohlc_map['Open']]
                    close_prices = df[ohlc_map['Close']]
                    high_prices = df[ohlc_map['High']]
                    low_prices = df[ohlc_map['Low']]
                    
                    # Cálculo do Relative Vigor Index
                    rvi = relative_vigor_index(open_prices, close_prices, high_prices, low_prices, period)
                    
                    # Cálculo do sinal
                    signal = pd.Series(0, index=rvi.index)
                    signal[rvi > rvi.shift(1)] = 1  # Sinal de compra
                    signal[rvi < rvi.shift(1)] = -1  # Sinal de venda
                
                    
                    # new_features[f'RVI_{period}'] = rvi
                    new_features[f'RVI_Signal_{period}'] = signal
                    # new_features[f'RVI_MA_{period}'] = rvi_ma
                    
                    logging.info(f"Relative Vigor Index with period {period} created successfully")
                except Exception as e:
                    logging.error(f"Error calculating Relative Vigor Index for period {period}: {str(e)}")
        
        return new_features
            
                  
    
########################################################################################
#                     4)  Cálculo Features janeladas Felipe 
########################################################################################

"""
Features janeladas do Felipe ( ROC, MACD_hist e PPO_hist)

"""

def calculate_indicators(df, h, window=100)-> DataFrame:
    df_copy = df.copy()
    results = {'Date': df_copy['Date'].values}

    print(f"Calculating indicators for h={h}")
    
    results[f'ROC_{h}'] = [np.nan for _ in range(window)]
    results[f'MACD_hist_{h}'] = [np.nan for _ in range(window)]
    results[f'PPO_hist_{h}'] = [np.nan for _ in range(window)]
    
    for i in range(window, df_copy.shape[0]):
        
        try:
            data_window = df_copy.iloc[i-window:i+1, :].copy()
            
            #ROC
            results[f'ROC_{h}'].append(clib_features.roc(data_window['Close'], length=h).iloc[-1])
            
            # macd, apenas o MACD_hist
            macd = clib_features.macd(data_window['Close'], fast_period=h, slow_period=h * 2, signal_period=h // 2)
            macd_col_name_h = f'MACDh_{h}_{h*2}_{h//2}'
            macd_hist =  macd[macd_col_name_h] 
            results[f'MACD_hist_{h}'].append(macd_hist.iloc[-1])
            
            # PPO, apenas o PPO_hist
            ppo = clib_features.ppo(data_window['Close'], fast=h, slow=h * 2)
            ppo_hist = ppo.iloc[:,2]
            results[f'PPO_hist_{h}'].append(ppo_hist.iloc[-1])
            
        except Exception as e:
              print(f'Erro no cálculo das features:{e} !')
            
            

    return pd.DataFrame(results)

import numpy as np
import pandas as pd
def process(data: pd.DataFrame):
    df = data.copy()
    
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['month'] = df['trans_date_trans_time'].dt.month
    df["day_of_week"] = df['trans_date_trans_time'].dt.dayofweek
    df["is_weekend"] = (df['trans_date_trans_time'].dt.dayofweek > 4).astype(int)
    df['hour'] = (df['trans_date_trans_time'].dt.hour).astype(int)
    
    df['num_of_trans'] = df['cc_num'].map(df.groupby('cc_num')['merchant'].count())
    df['num_of_unique_merchant'] = df['cc_num'].map(df.groupby('cc_num')['merchant'].nunique())
    
    df['lat_std'] = df['cc_num'].map(df.groupby('cc_num')['lat'].std())
    df['long_std'] = df['cc_num'].map(df.groupby('cc_num')['long'].std())
    
    df['country'] = 'United States'
    df['mean_amt_per_category'] = df['category'].map(df.groupby('category')['amt'].mean())
    
    allLat  = list(df['lat']) + list(df['merch_lat'])
    medianLat  = sorted(allLat)[int(len(allLat)/2)]
    latMultiplier  = 111.32

    df['lat'] = latMultiplier  * (df['lat']   - medianLat)
    df['merch_lat']   = latMultiplier  * (df['merch_lat']  - medianLat)
    allLong = list(df['long']) + list(df['merch_long'])

    medianLong  = sorted(allLong)[int(len(allLong)/2)]

    longMultiplier = np.cos(medianLat*(np.pi/180.0)) * 111.32
    df['long']  = longMultiplier * (df['long']  - medianLong)
    df['merch_long']  = longMultiplier * (df['merch_long'] - medianLong)


    df['long_diff'] = df['merch_long'] - df['long']
    df['lat_diff'] = df['merch_lat'] - df['lat']

    df['distance_km'] = (df['long_diff']**2 + df['lat_diff']**2)**(1/2)
    
    df['age'] = (pd.to_datetime(df['unix_time'], unit='s') - pd.to_datetime(df['dob'])) / pd.Timedelta(days=365.25)
    df['age'] = df['age'].round()
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})
    cols_mean_target = ['city', 'state', 'job']
    for col in cols_mean_target:
        if df[col].nunique() < 10:
            one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat((df.drop(col, axis=1), one_hot), axis=1)
        
        else:
            mean_target = df.groupby(col)['is_fraud'].mean()
            df[col] = df[col].map(mean_target)
    df['amt_above_mean'] = df['amt'] - df.groupby('cc_num')['amt'].mean()
    df['category'] = df['category'].map(df.groupby('category')['is_fraud'].mean())
    df = df.drop(['first', 'last', 'trans_date_trans_time', 'street', \
                 'unix_time', 'dob', 'trans_num', 'long_diff', 'lat_diff', 'merchant', 'cc_num', 'zip'], axis=1)
    return df

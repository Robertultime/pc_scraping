import pandas as pd
import numpy as np
import datetime
import os
import re
import scrap


def list_files(path):
    files = []
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):
            filename = datetime.datetime.strptime(filename.replace('.csv',''), "%Y-%m-%d")
            files.append(filename)
    return files


def load_last_df(path='pc_dataframe'):
    
    f_list = list_files(path)
    
    last_file = sorted(f_list)[-1]
    
    df = pd.read_csv(path + '/'+str(last_file.date())+'.csv', index_col = 0)
    
    return df

### Cleaning of the datas

def split_res(res):
    end = res.index('pix')
    chaine = res[:res.index('pix')].replace('Résolution ', "")
    match = re.search(r'x .', chaine)
    if match:
        index = match.start() - 5
        partie_avant_nombre = chaine[:index]
        partie_apres_nombre = chaine[index:]
        return partie_avant_nombre.strip(), partie_apres_nombre.strip()
    else:
        return chaine, ''
    
def split_ram(ram):
    pattern = r'( Go | - | MHz)'
    ram_split = re.split(pattern, ram)
    return [int(ram_split[0]),ram_split[2],int(ram_split[4])]

def split_mem(mem):
    pattern = r'( Go | To | \+ )'
    
    return re.split(pattern, mem)

def get_mem(mem):
    
    if mem[1] == ' To ':
        mem[0] *= 1000
    if mem[5] == ' To ':
        mem[4] *= 1000
    mem[0] += mem[4]
    
    return mem

def df_exp(df):
    df[['Res_qual','Res_px']] = pd.DataFrame(list(df['Res'].apply(split_res)))

    df[['RAM_Go','RAM_tech','RAM_frq']] = pd.DataFrame(list(df['RAM'].apply(split_ram)))
    
    m_list = pd.Series(df['Memory'])

    df_split = pd.DataFrame(list(m_list.apply(split_mem)))
    df_split[0] = df_split[0].astype('int')
    df_split[4] = df_split[4].fillna(0).astype(int)
    df_split = df_split.apply(get_mem, axis=1)[[0,2]]
    df_split[2] = df_split[2].str.replace('(','').str.replace(')','')
    df[['Mem_Go','Mem_tech']] = df_split
    
    df['GPU_brand'] = df['GPU'].str.split(' ', expand=True).iloc[:,0]
    
    return df


def GPU_clean(df):
    
    
    df['GPU_c'] = df['GPU'].str.split(' ', expand=True).iloc[:,1:].fillna('').agg(' '.join, axis=1)

    df['GPU_c'] = df['GPU_c'].str.replace('GeForce','')
    df.GPU_c = df['GPU_c'].str.split('Version', expand=True).iloc[:,0]
    
    
    df['GPU_c'] = df.GPU_c.str.strip()
    
    cleaning_dic = {
        'GTX 1650 Max-Q(4 Go)':'GeForce GTX 1060 with Max-Q Design',
        'GTX 1650 Ti(4 Go)':'GeForce GTX 1650 Ti',
        'GTX 1650 Ti Max-Q(4 Go)':'GeForce GTX 1650 Ti with Max-Q Design',
        'GTX 1650(4 Go)':'GeForce GTX 1650 (Mobile)',
        'RTX 3050(4 Go)': 'GeForce RTX 3050',
        'RTX 3050 Ti Max-Q(4 Go)': 'GeForce RTX 3050 OEM',
        'RTX 3050(6 Go)': 'GeForce RTX 3050 6GB Laptop GPU',
        'RTX 3050 Max-Q(4 Go)': 'GeForce RTX 3050 Laptop GPU',
        'RTX 3050 Ti(4 Go)': 'GeForce RTX 3050 Ti Laptop GPU',
        'RTX 3060 Max-P(6 Go)' : 'GeForce RTX 3060 Laptop GPU', 
        'RTX 3060 Max-Q(6 Go)' : 'GeForce RTX 3060 Laptop GPU', 
        'RTX 3060 Max-Q(8 Go)' : 'GeForce RTX 3060 Ti',
        'RTX 3070 Max-Q(8 Go)':'GeForce RTX 3070 Laptop GPU',
        'RTX 3070 Max-P(8 Go)':'GeForce RTX 3070 Laptop GPU',
        'RTX 3070 Ti Max-P(8 Go)':'GeForce RTX 3070 Ti Laptop GPU',
        'RTX 3070 Ti Max-Q(8 Go)':'GeForce RTX 3070 Ti Laptop GPU',
        'RTX 3080(16 Go)':'GeForce RTX 3080 Ti Laptop GPU',
        'RTX 3080 Ti Max-Q(16 Go)':'GeForce RTX 3080 Ti Laptop GPU',
        'RTX 3080 Max-P(8 Go)':'GeForce RTX 3080 Laptop GPU',
        'RTX 3080 Max-Q(8 Go)':'GeForce RTX 3080 Laptop GPU',
        'RTX 3080 Ti Max-P(16 Go)':'GeForce RTX 3080 Ti Laptop GPU',
        'RTX 3080 Ti Max-P(8 Go)':'GeForce RTX 3080 Ti Laptop GPU',
        'RTX 3080 Max-P(16 Go)':'GeForce RTX 3080 Ti Laptop GPU',
        'RTX 3080 Max-Q(16 Go)':'GeForce RTX 3080 Ti Laptop GPU',
        'RTX 4050 (Mobile)(6 Go)':'GeForce RTX 4050 Laptop GPU',
        'RTX 4050 (Mobile)(8 Go)':'GeForce RTX 4050 Laptop GPU',
        'RTX 4050 (Mobile)(4 Go)':'GeForce RTX 4050 Laptop GPU',
        'RTX 4060 (Mobile)(8 Go)':'GeForce RTX 4060 Laptop GPU',
        'RTX 4060(8 Go)':'GeForce RTX 4060 Laptop GPU', 
        'RTX 4060 (Mobile)(6 Go)':'GeForce RTX 4060 Laptop GPU',
        'RTX 4070 (Mobile)(8 Go)':'GeForce RTX 4070 Laptop GPU',
        'RTX 4080 (Mobile)(12 Go)':'GeForce RTX 4080 Laptop GPU',
        'RTX 4090 (Mobile)(16 Go)':'GeForce RTX 4090 Laptop GPU',
        'Radeon RX 6650M(8 Go)':'Radeon RX 6650M',
        'Radeon RX 6700S(8 Go)':'Radeon RX 6700S',
        'Radeon RX 6500M(4 Go)':'Radeon RX 6650M',
        'Radeon RX 6800M(12 Go)':'Radeon RX 6800M',
        'Radeon RX 6600M(8 Go)':'Radeon RX 6600M',
        'Radeon RX 6850M XT(12 Go)':'Radeon RX 6850M XT',
        'RTX 2050(4 Go)': 'GeForce RTX 2050',
        'RTX A500(4 Go)':'RTX A500 Laptop GPU',
        'RTX A2000(8 Go)':'RTX A2000 8GB Laptop GPU',
        'RTX 3500 ADA(12 Go)':'RTX 3500 Ada Generation Laptop GPU',
        'RTX A2000(4 Go)':'RTX A2000 Laptop GPU',
        'RTX 2000 ADA(8 Go)':'RTX A2000 8GB Laptop GPU',
        'RTX A1000(6 Go)':'RTX A1000 6GB Laptop GPU',
        'RTX A3000(8 Go)':'RTX A3000 Laptop GPU',
        'RTX A4000(12 Go)':'RTX A4000 Laptop GPU',
        'RTX 5000 ADA(16 Go)':'RTX 5000 Ada Generation Laptop GPU',
        'RTX 3000 ADA(8 Go)':'RTX 3000 Ada Generation Laptop GPU',
        'RTX A4500(16 Go)':'RTX A4500 Laptop GPU',
        'RTX A3000(12 Go)':'RTX A3000 12GB Laptop GPU',
        'RTX A5000(16 Go)':'RTX A5000 Laptop GPU',
        'RTX A1000(4 Go)':'RTX A1000 Laptop GPU',
        'RTX A5500(16 Go)':'RTX A5500 Laptop GPU',
        'RTX A3000(6 Go)':'RTX A3000 Laptop GPU',
        'Quadro RTX 5000(16 Go)':'Quadro RTX 5000 (Mobile)',
        'RTX 2070 Super Max-Q(8 Go)':'GeForce RTX 2070 Super with Max-Q Design',
        'RTX 2070 Super(8 Go)':'GeForce RTX 2070 (Mobile)',
        'Quadro RTX 4000(8 Go)':'Quadro RTX 4000 (Mobile)',
        'RTX 2080 Max-Q(8 Go)':'GeForce RTX 2080 with Max-Q Design',
        'Adreno 618':np.nan, 
        'Adreno 690':np.nan, 
        'Adreno 685':np.nan,
        'Arc A370M(4 Go)':'Intel Arc A370M', 
        'Arc A350M(4 Go)':np.nan,
        'Arc A370M(12 Go)':'Intel Arc A370M',
        'Iris Xe Graphics (96 EU) - Raptor Lake':'Intel Iris Xe',
        'Iris Xe Graphics (80 EU) - Raptor Lake':'Intel Iris Xe',
        'Iris Xe Graphics (96 EU)':'Intel Iris Xe',
        'Iris Xe Graphics (80 EU)':'Intel Iris Xe',
        'UHD Graphics (16 EU)':np.nan,
        'UHD Graphics (23 EU)':np.nan,
        'UHD Graphics (24 EU)':np.nan,
        'UHD Graphics (32 EU)':np.nan,
        'UHD Graphics (48 EU) - Alder Lake':np.nan,
        'UHD Graphics (48 EU) - Raptor Lake':np.nan,
        'UHD Graphics (48 EU) - Tiger Lake':np.nan,
        'UHD Graphics (64 EU) - Alder Lake':np.nan,
        'UHD Graphics (64 EU) - Raptor Lake':np.nan,
        'UHD Graphics 600':np.nan,
        'UHD Graphics 605':np.nan,
        'UHD Graphics 610':np.nan,
        'UHD Graphics 620':np.nan,
        'UHD Graphics G1 (32 EU)':np.nan,
        'T1200(4 Go)':'T1200 Laptop GPU',
        'T550(4 Go)':'T550 Laptop GPU',
        'T600(4 Go)':'T600 Laptop GPU',
        'MX350(2 Go)':'GeForce MX350',
        'MX450(2 Go)':'GeForce MX450',
        'MX550(2 Go)':'GeForce MX550',
        'Radeon RX 6650M(4 Go)':'Radeon RX 6650M',
        'Radeon RX 7600S(8 Go)':'Radeon RX 7600S',
        'Quadro P520(4 Go)':np.nan,
        'Quadro T1200(4 Go)':'T1200 Laptop GPU',
        'Radeon 660M':'Radeon 660M Ryzen 5 7535HS',
        'Radeon 680M':'Radeon 680M Ryzen 7 7735HS',
        
    }
    
    df['GPU_c'] = df['GPU_c'].map(cleaning_dic)
    


def CPU_clean(df):
    
    df['CPU_c'] = df['CPU'].str.split('(', expand=True)[0].str.strip()
    
    
    cleaning_dic = {
        'AMD Ryzen 5 7520U(4 coeurs 4.30 GHz)':'AMD Ryzen 5 7520U',
        'AMD Ryzen 7 Pro 6850H(8 coeurs 4.70 GHz)':'AMD Ryzen 7 PRO 6850H',
        'AMD Ryzen 5 4600H(6 coeurs 4.00 GHz)': 'AMD Ryzen 5 4600H',
        'AMD Ryzen 5 7530U(6 coeurs 4.50 GHz)': 'AMD Ryzen 5 7530U',
        'AMD Ryzen 5 5500H(4 coeurs 4.20 GHz)': 'AMD Ryzen 5 5500H',
        'AMD Ryzen 5 5625U(6 coeurs 4.30 GHz)': 'AMD Ryzen 5 5625U',
        'AMD Ryzen 5 5500U(6 coeurs 4.00 GHz)': 'AMD Ryzen 5 5500U',
        'AMD Ryzen 5 7535HS(6 coeurs 4.55 GHz)': 'AMD Ryzen 5 7535HS',
        'AMD Ryzen 5 6600H(6 coeurs 4.50 GHz)': 'AMD Ryzen 5 6600H',
        'AMD Ryzen 5 5600H(6 coeurs 4.10 GHz)': 'AMD Ryzen 5 5600H',
        'AMD Ryzen 5 5600U(6 coeurs 4.20 GHz)': 'AMD Ryzen 5 5600U',
        'AMD Ryzen 5 6600U(6 coeurs 4.50 GHz)': 'AMD Ryzen 5 6600U',
        'AMD Ryzen 5 7535U(6 coeurs 4.55 GHz)': 'AMD Ryzen 5 7535U',
        'AMD Ryzen 5 7520C(4 coeurs 4.30 GHz)': 'AMD Ryzen 5 7520C',
        'AMD Ryzen 5 7640HS(6 coeurs 5.00 GHz)': 'AMD Ryzen 5 7640HS',
        'AMD Ryzen 5 Pro 6650U(6 coeurs 4.50 GHz)': 'AMD Ryzen 5 Pro 6650U',
        'AMD Ryzen 5 4500U(6 coeurs 4.00 GHz)': 'AMD Ryzen 5 4500U',
        'AMD Ryzen 5 3500U(4 coeurs 3.70 GHz)': 'AMD Ryzen 5 3500U',
        'AMD Ryzen 5 Pro 5675U(6 coeurs 4.30 GHz)': 'AMD Ryzen 5 Pro 5675U',
        'AMD Ryzen 5 3450U(4 coeurs 3.50 GHz)': 'AMD Ryzen 5 3450U',
        'AMD Ryzen 5 Pro 5650U(6 coeurs 4.20 GHz)': 'AMD Ryzen 5 Pro 5650U',
        'AMD Ryzen 5 Pro 6650U':'AMD Ryzen 5 PRO 6650U',
        'AMD Ryzen 5 Pro 5675U':'AMD Ryzen 5 PRO 5675U',
        'AMD Ryzen 5 Pro 5650U':'AMD Ryzen 5 PRO 5650U',
        'AMD Ryzen 3 3250C': 'AMD Ryzen 3 3250C',
        'AMD Ryzen 3 3250U': 'AMD Ryzen 3 3250U',
        'AMD Ryzen 7 Pro 6850H': 'AMD Ryzen 7 PRO 6850H',
        'AMD Ryzen 7 Pro 6850U': 'AMD Ryzen 7 PRO 6850U',
        'AMD Ryzen 5 7520C': 'AMD Ryzen 5 7520C',
        'AMD Ryzen 9 7945HX3D': 'AMD Ryzen 9 7945HX3D',
        'AMD Ryzen 7 3700U': 'AMD Ryzen 7 3700U',
        'AMD Ryzen 9 6980HX': 'AMD Ryzen 9 6980HX',
        'AMD Ryzen 5 3500U': 'AMD Ryzen 5 3500U',
        'AMD Ryzen 5 3450U': 'AMD Ryzen 5 3450U',
        'AMD Ryzen 7 4980U': 'AMD Ryzen 7 4980U',
    }
    
    df['CPU_c'] = df['CPU_c'].map(cleaning_dic).fillna(df['CPU_c'])
    
    df['CPU_c'] = df['CPU_c'].str.replace('Pro', 'PRO')
    
def clean_df(df):
    GPU_clean(df)
    CPU_clean(df)


# Datas notation

def min_max(sr):
    return (sr-sr.min())/(sr.max()-sr.min())

def comp_notation(df, comp):
    return min_max(df.groupby(comp).median()['Price'].sort_values())

def comps_dic(df, comp_list):
    dic = {}
    for comp in comp_list:
        dic[comp] = comp_notation(df, comp)
    return dic

def notation(pc, Comp_notations, factors):
    note = 0
    for fact in factors.index:
        note += Comp_notations[fact][pc[fact]] * factors[fact]
    return note

def df_noted(df, factors):
    
    factors = factors/factors.sum()
    
    Comp_notations = comps_dic(df, list(factors.index))
    df['Note'] = df.apply(lambda x: notation(x, Comp_notations, factors), axis=1)

    df['Adj_Note'] = (df['Note'] / df['Price'])*1000
    
    return df

#permet de noter les composants continus
def note_function(x, alpha, beta):
    return 1/(1+np.exp(-x/alpha+beta))

def df_noted2(df, factors):
    
    factors = factors/factors.sum()
    df['Note_Finale'] = 0
    for factor in factors.index:
        df['Note_Finale'] += df[factor] * factors[factor]
        
    df['Note_diff'] = df['Note_Finale']-df['Price'].apply(simple_model)
        
    
        
def simple_model(x):
    a1, b1 = -0.33146807, 0.00105002
    a2, b2 = 0.31462099, 0.00020849
    a3, b3 = 0.88924409, 1.38216007e-05

    s1 = 800
    s2 = 3000
    
    if x<= s1:
        return a1 + b1*x
    if s2 < x:
        return a3 + b3*x
    else :
        return a2 + b2*x
    


def note_df(df):

    CPU_df = load_last_df('CPU_bench')
    GPU_df = load_last_df('GPU_bench')
    
    clean_df(df)
    df = df_exp(df)
    CPU_df['Name'] = CPU_df['Name'].str.split('@', expand=True)[0].str.strip()

    p1 = pd.merge(df, GPU_df, left_on='GPU_c', right_on='Name', suffixes=('', '_GPU')).drop(['Name_GPU', 'Price ($)', 'URL', 'GPU_c'], axis=1)
    p2 = pd.merge(p1, CPU_df, left_on='CPU_c', right_on='Name', suffixes=('', '_CPU'), how='left').drop(['Name_CPU', 'Price ($)', 'URL', 'CPU_c'], axis=1)

    p2 = p2.rename({'Note':'Note_GPU' }, axis=1)
    p2['Note_GPU'] = min_max(p2['Note_GPU'])
    p2['Note_CPU'] = min_max(p2['Note_CPU'])
    
    p2['Note_Res'] = min_max(p2['Res_px'].str.split(' x ', expand=True).astype(int).product(axis=1).apply(np.log))
    
    p2['Note_Mem'] = p2['Mem_Go'].apply(lambda x:note_function(x,170, 2.5))
    p2['Note_RAM'] = p2['RAM_Go'].apply(lambda x:note_function(x,7, 2.5))
    
    factors = pd.Series(
    {'Note_GPU':20,
     'Note_CPU':5, 
     'Note_RAM':2, 
     'Note_Mem':2,
     'Note_Res':2})
    
    df_noted2(p2, factors)
    
    return p2

def get_pc_noted():
    pc_df = load_last_df()
    pc_df = note_df(pc_df)
    
    return pc_df

def get_best_pc(pc_noted):
    return pc_noted[(pc_noted.Note_Finale > 0.5) & (pc_noted.Price <= 1500 | (pc_noted.Note_diff>=0.02)) 
                    & (pc_noted.Note_diff>=0.02)].sort_values('Note_diff', ascending=False)

def computer_check():
    scrap.update_df()
    scrap.update_bench()
    pc_noted = get_pc_noted()
    return get_best_pc(pc_noted)
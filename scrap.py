from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import datetime


def get_soup(url):
    
    try:
        response = requests.get(url)
    except Exception as e:
        print(e)
        
    if not response.status_code == 200:
        print(f"La requête n°{i} a échoué avec le code de statut :", response.status_code)
    if response:
        soup = BeautifulSoup(response.text, 'html.parser')
    return soup
    
#Recherche de la dernière page du site
def get_last_page(url):
    
    soup = get_soup(url)

    # Trouver l'élément <a> avec la classe "page-link"
    a_element = soup.find_all('a', class_='nojs', attrs={"aria-label": "Next"})

    # Extraire la valeur de l'attribut "rel"
    rel_value = list(a_element)[1]['rel']

    # Convertir la valeur en int
    rel_int = int(rel_value[0])

    # Imprimer la valeur en int
    return(rel_int)


def get_pc_dic(div):
    if div.find('a', class_="btn btn-md btn-squared btn-success price go"):
        title = div.find('a', class_='white').text
        price = float(div.find('a', class_="btn btn-md btn-squared btn-success price go").text.replace(' ','').replace('€','').replace(',','.'))
        img_url = div.find('img')['src']
        seller = div.find_all('img')[1]['alt']
        pc_url = "https://www.comparez-malin.fr" + div.find('a', class_='white')['href']

        components = div.find('ul').find_all('li')

        if len(components) == 8:
            Screen = np.nan
            Res = np.nan
            Eth = components[6].text.strip().replace('\xa0','').replace('x\n','')
            Wifi = components[7].text.strip().replace('\xa0','').replace('x\n','')
            
        elif len(components) == 10:
            Screen = components[0].text.strip().replace('\xa0','').replace('x\n','')
            Res = components[1].text.strip().replace('\xa0','').replace('x\n','')
            components = components[2:]
            Eth = components[6].text.strip().replace('\xa0','').replace('x\n','')
            Wifi = components[7].text.strip().replace('\xa0','').replace('x\n','')
            
        elif len(components) == 9:
            Screen = components[0].text.strip().replace('\xa0','').replace('x\n','')
            Res = components[1].text.strip().replace('\xa0','').replace('x\n','')
            components = components[2:]
            Eth = np.nan
            Wifi = components[6].text.strip().replace('\xa0','').replace('x\n','')
            
        elif len(components) == 7:
            Screen = np.nan
            Res = np.nan
            Eth = components[6].text.strip().replace('\xa0','').replace('x\n','')
            Wifi = np.nan
            

        CPU = components[0].text.strip().replace('\xa0','').replace('x\n','')
        GPU = components[1].text.strip().replace('\xa0','').replace('x\n','')
        RAM = components[2].text.strip().replace('\xa0','').replace('x\n','')
        Memory = components[3].text.strip().replace('\xa0','').replace('x\n','')
        OS = components[4].text.strip().replace('\xa0','').replace('x\n','')
        USB = int(components[5].text.strip().replace('\xa0','').replace('x\n','').replace(" x USB",""))


        dic = {"Screen":Screen,
          "Res":Res,
          "CPU":CPU,
          "RAM":RAM,
          "Memory":Memory,
          "OS":OS,
          "USB":USB,
          "Eth":Eth,
          "Wifi":Wifi,
          "Title":title,
          "Price":price,
          "Img_url":img_url,
          "Seller":seller,
          "PC_url":pc_url}

        return(dic)
    else:
        return None

    
#permet de transformer le div initial en dictionnaire avec les informations
def get_portable_dic(div):
    if div.find('a', class_="btn btn-md btn-squared btn-success price go"):
        name = div.find('a', class_='white').text
        price = float(div.find('a', class_="btn btn-md btn-squared btn-success price go").text.replace(' ','').replace('€','').replace(',','.'))
        img_url = div.find('img')['src']
        seller = div.find_all('img')[1]['alt']
        pc_url = "https://www.comparez-malin.fr" + div.find('a', class_='white')['href']

        components = div.find('ul').find_all('li')
            
        
        Screen = components[0].text.strip().replace('\xa0','').replace('x\n','')
        Res = components[1].text.strip().replace('\xa0','').replace('x\n','')
        OS = components[2].text.strip().replace('\xa0','').replace('x\n','')
        CPU = components[3].text.strip().replace('\xa0','').replace('x\n','')
        GPU = components[4].text.strip().replace('\xa0','').replace('x\n','')
        RAM = components[5].text.strip().replace('\xa0','').replace('x\n','')
        Memory = components[6].text.strip().replace('\xa0','').replace('x\n','')
        Wifi = components[7].text.strip().replace('\xa0','').replace('x\n','')


        dic = {"Name":name,
            "Screen":Screen,
            "Res":Res,
            "Price":price,
            "GPU":GPU,
            "CPU":CPU,
            "RAM":RAM,
            "Memory":Memory,
            "OS":OS,
            "Wifi":Wifi,
            "Seller":seller,
            "PC_url":pc_url,
            "Img_url":img_url,
}

        return(dic)
    else:
        return None

#Permet d'obtenir un DataFrame d'information sur chaque pc
def get_pc_df():
    n_page = get_last_page("https://www.comparez-malin.fr/informatique/pc-portable")
    dic_list = []

    for i in range(1, n_page+1):
        
        if i%50 == 0:
            print(i)

        url = "https://www.comparez-malin.fr/informatique/pc-portable/?page=" + str(i)

        soup = get_soup(url)
        
        if soup:
            divs = soup.find_all('div', class_='col-xs-12 col-sm-12 col-md-12 col-lg-6 col-xl-4 col-xxl-4 m-b-10')

            for div in divs : 
                if get_portable_dic(div):
                    dic_list.append(get_portable_dic(div))

    pc_df = pd.DataFrame(dic_list)
    
    return pc_df


def save_df(df, path='pc_dataframe'):
    name = str(datetime.datetime.now().date())+'.csv'
    df.to_csv(path + '/' + name)
    return


def update_df():
    df = get_pc_df()
    save_df(df)
    return

    
def update_bench():
    CPU_df = get_bench_df('https://www.cpubenchmark.net/high_end_cpus.html')
    GPU_df = get_bench_df('https://www.videocardbenchmark.net/high_end_gpus.html')
    
    save_df(CPU_df,'CPU_bench')
    save_df(GPU_df,'GPU_bench')
    return

    
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

def get_bench_df(url):
    soup = get_soup(url)
    
    url = 'https://' + url.split('/')[2]
    
    
    li_tags = soup.find('ul', class_='chartlist').find_all('li')

    # Parcourir chaque balise 'li' et extraire les informations nécessaires

    dic_list = []
    for li_tag in li_tags:
        prdname = li_tag.find('span', class_='prdname').get_text()
        count = li_tag.find('span', class_='count').get_text()
        price_neww = li_tag.find('span', class_='price-neww').get_text()
        href = li_tag.find('a')['href']

        dic = {'Name':prdname,
              'Note':int(count.replace(',','')),
              'Price ($)':float(price_neww.replace(',','').replace('*','').replace('$','').replace('NA','0')),
              'URL':url+'/'+href}

        dic_list.append(dic)

    df = pd.DataFrame(dic_list)
    return df

if __name__ =="__main__":
    update_df()
    update_bench()
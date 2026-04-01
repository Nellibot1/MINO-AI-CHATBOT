import os
import re
import requests
import datetime
import pdfplumber
from turtle import pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sqlalchemy import create_engine, Table, MetaData, Column, Text, Integer, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker
from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Verbindung zur Datenbank herstellen
engine = create_engine('mysql+mysqlconnector://mino:XXXX')

# Erstellt eine Session
Session = sessionmaker(bind=engine)
session = Session()

metadata = MetaData()

scraping_table = Table('Scraping', metadata,
    Column('Scraping_ID', Integer, primary_key=True, autoincrement=True),
    Column('url', Text),
    Column('Prompt', Text),
    Column('Inhalt', Text),
    Column('Zuletzt_gescraped', DateTime, default=func.now()),
)

# Funktion zum Scrapen der AM und HdM-Seite
def scrape_links(base_url, additional_links, url_blacklist, word_blacklist):
    data_hdm = [] 
    scraped_urls = set()  
    scraped_elements = {}  
    scraped_contents = set()

    response = requests.get(base_url)
    response.encoding = 'ISO-8859-1'  
    soup = BeautifulSoup(response.content, 'html.parser')

    links = soup.find_all("a") + [BeautifulSoup('<a href="{}"></a>'.format(link), 'html.parser').a for link in additional_links]

    for link in links:
        absolute_link = urljoin(base_url, link.get("href"))

        if absolute_link in url_blacklist or absolute_link.startswith("javascript:") or absolute_link.startswith("tel:") or absolute_link in scraped_urls:
            continue

        scraped_urls.add(absolute_link)

        link_response = requests.get(absolute_link)
        link_soup = BeautifulSoup(link_response.content, 'html.parser')

        content = {'div': [], 'p': [], 'table': [], 'li': [], 'ul': [], 'h1': [], 'h2': [], 'h3': [], 'title': []}
        for tag_name in content.keys():
            tags = link_soup.find_all(tag_name)

            for tag in tags:
                if any(blacklisted_word in tag.text for blacklisted_word in word_blacklist) or \
                    tag.find_parent("header") or tag.find_parent("footer") or \
                    tag.get('class') and any(blacklisted_word in tag.get('class') for blacklisted_word in word_blacklist) or \
                    tag.get('id') and any(blacklisted_word in tag.get('id') for blacklisted_word in word_blacklist) or \
                    tag.find_parent(id=lambda id: id and any(blacklisted_word in id for blacklisted_word in word_blacklist)) or \
                    tag.find_parent(class_=lambda cls: cls and any(blacklisted_word in cls for blacklisted_word in word_blacklist)):
                    continue
                
                tag_text = tag.text.strip().split('\n\n\n')

                for sub_tag_text in tag_text:
                    sub_tag_text = sub_tag_text.strip()
                    if sub_tag_text not in scraped_contents:  
                        if sub_tag_text in scraped_elements:
                            content[tag_name].append('Siehe [{}]'.format(scraped_elements[sub_tag_text]))
                        else:
                            content[tag_name].append(' '.join(sub_tag_text.split()))
                            scraped_elements[sub_tag_text] = absolute_link

                        scraped_contents.add(sub_tag_text)

        prompt = ' '.join(content['h1'] + content['h2'] + content['h3'] + content['title']).replace("Hochschule der Medien", "")
        if "ein_modul_ajax" in absolute_link:
            url = f'URL Studienverlaufsplan: https://www.hdm-stuttgart.de/am/studiengang/studienverlaufsplan : '
            data_hdm.append((f'https://www.hdm-stuttgart.de/am/studiengang/studienverlaufsplan', prompt, url + ' '.join([' '.join(sublist) for sublist in content.values()])))
        else:
            joined_content = ' '.join([' '.join(sublist) for sublist in content.values()])
            if joined_content.strip():
                url = f'URL: {absolute_link} : '
                joined_content_with_url = url + joined_content
                data_hdm.append((absolute_link, prompt, joined_content_with_url))

    return data_hdm

# Funktion zum Scrapen des StarPlan 
def splan_scraping(word_blacklist):
    url = 'https://splan.hdm-stuttgart.de/splan/json?m=getros&loc=1'

    response = requests.get(url)
    data = response.json()
    room_ids = []
    data_splan = []

    for entry in data:
        for item in entry:
            room_ids.append(item['id'])
            
    base_url = 'https://splan.hdm-stuttgart.de/splan/json?m=getTT&sel=ro&pu=34&ro={}&sd=false&loc=1&sa=false&cb=o'

    for id in room_ids:
        url = base_url.format(id)  
        response = requests.get(url)
        response.encoding = 'ISO-8859-1'
        soup = BeautifulSoup(response.content, 'lxml')

        weekdays = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
        last_sorting_time = None
        current_weekday_index = 0
        splan_entry = ''
        events = ''

        room_shortname = [item['shortname'] for item in data[0] if item['id'] == id][0]
        room_name = [item['name'] for item in data[0] if item['id'] == id][0]
        room_name = '' if room_name == room_shortname else f'({room_name})'

        tags = soup.find_all('div', class_='tooltip')

        for tag in tags:
            if any(blacklisted_word in tag.text for blacklisted_word in word_blacklist):
                continue
            
            time = ''.join(re.findall('\d{2}:\d{2}(?:-\d{2}:\d{2})?', tag.get_text()))
            start_time = time.split('-')[0]
            start_time_hour, start_time_minutes = start_time.split(':')
            sorting_time = int(start_time_hour + start_time_minutes)
            if last_sorting_time is None or sorting_time < last_sorting_time:
                day = weekdays[current_weekday_index]
                current_weekday_index = (current_weekday_index + 1) % len(weekdays)
            last_sorting_time = sorting_time

            dates = re.findall('\d{2}\.\d{2}\.\d{4}', tag.get_text())
            date = dates[1] if len(dates) >= 2 else (dates[0] if dates else 'wöchentlich')
            if date != 'wöchentlich':
                date_obj = datetime.datetime.strptime(date, '%d.%m.%Y')
                day = weekdays[date_obj.weekday()]

            event = str(tag).split('>', 1)[1].split('<br/>', 1)[0]
            if event.strip() not in events.strip(' , '): events += f"{''.join(char for char in event if char.isalnum() or char.isalpha() or char.isspace() or char == '-')} , "
            degree = ', '.join(re.findall('(?:Aka|CLnD|GD|HoRads|KS-BI|Minors|Reservierungen|Prüfungen|Projekte|Sprachenangebot|Sonst\. Abt\.|Startup|Sitzungen|VC|Verlegung|VS|Veranstaltungen|(?:[A-Z]{2,3}[0-9]{1,2}(?:-[0-9])?|[A-Z]{3}[0-9]?|[A-Za-z]{3}[0-9]-[0-9]))', tag.text))[1:]
            room_shortname = room_shortname
            room_name = room_name
            dozent = str(tag).split('<br/>'+room_shortname)[0].split('<br/>')[-2]
            if 'div' in dozent: dozent = ''
            module_numbers = re.findall('\(\d+\w?\)', event)
            module_link = ', '.join(f"https://www.hdm-stuttgart.de/vorlesung_detail_edvnr?edvnr={module_number[1:-1]}" for module_number in module_numbers)
            splan_entry += f'{day} {date}: {time} "{event}" Raum "{room_shortname} {room_name}", Studiengang "{degree}", Veranstalter "{dozent}", Link "{module_link}"\n'

        prompt = f'Raum: {room_shortname} ; Veranstaltungen: {events.rstrip(" , ")}'
        if splan_entry: 
            data_splan.append((f'https://splan.hdm-stuttgart.de/splan/mobile?lan=de&acc=true&act=tt&sel=ro&pu=34&ro={id}&sd=false&loc=1&sa=false&cb=o', prompt, f'URL Raum {room_shortname}: https://splan.hdm-stuttgart.de/splan/mobile?lan=de&acc=true&act=tt&sel=ro&pu=34&ro={id}&sd=false&loc=1&sa=false&cb=o : ' + splan_entry)) 
        else: 
            data_splan.append((f'https://splan.hdm-stuttgart.de/splan/mobile?lan=de&acc=true&act=tt&sel=ro&pu=34&ro={id}&sd=false&loc=1&sa=false&cb=o', prompt, f'Keine Veranstaltungen, URL Raum  {room_shortname}: https://splan.hdm-stuttgart.de/splan/mobile?lan=de&acc=true&act=tt&sel=ro&pu=34&ro={id}&sd=false&loc=1&sa=false&cb=o')) 
                                                                                                                                                                                                                                     
    return data_splan

# Funktion zum Zusätzliche URLs Scrapen
def additional_url_scraping():
    additional_links= [
        "https://www.hdm-stuttgart.de/studieninteressierte/studium/bachelor",
        "https://www.hdm-stuttgart.de/studieninteressierte/studium/master",        
        "https://www.hdm-stuttgart.de/studieninteressierte/studium/weiterbildungsangebote",
        "https://vs-hdm.de/de/mitmachen/initiativen"
    ]
    additional_urls = []

    for additional_link in additional_links:

        additional_urls.append(additional_link)
            
        response = requests.get(additional_link)
        response.encoding = 'ISO-8859-1'  
        soup = BeautifulSoup(response.content, 'html.parser')
            
        links = soup.find_all("a")

        for link in links:
            absolute_link = urljoin(additional_link, link.get("href"))
            if ("studium/steckbrief?" in absolute_link or "initiativen/" in absolute_link) and absolute_link not in additional_urls:
                additional_urls.append(absolute_link)
    return additional_urls
    
# Funktion zum Generieren von URLs für Studienverlaufsplan-Scraping
def course_plan_scraping():
    url = "https://www.hdm-stuttgart.de/am/studiengang/studiengang/studienverlaufsplan/wahlpflicht_ajax"
    response = requests.get(url)
    module_ids = [int(num) for num in re.findall(r"openModulePopup\((\d+)\)", response.text)]
    course_plan_base_url = "https://www.hdm-stuttgart.de/am/studiengang/studienverlaufsplan/ein_modul_ajax?sgblockID={}"
    course_plan_urls = [course_plan_base_url.format(num) for num in module_ids]

    return course_plan_urls

# Funktion zum Scrapen der SPOs
def pdf_scraping(base_url):
    base_url = "https://www.hdm-stuttgart.de/studierende/studium/spo"
    data_pdfs = []
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    pdf_links = [urljoin(base_url, link.get('href')) for link in soup.find_all('a') if link.get('href', '').endswith('.pdf')]
    pdf_links = pdf_links[:2]
    save_dir = "files"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for pdf_link in pdf_links:
        pdf_response = requests.get(pdf_link)

        filename = pdf_link.split('/')[-1]

        pdf_path = os.path.join(save_dir, filename)
        with open(pdf_path, 'wb') as f:
            f.write(pdf_response.content)

        pdf_content = ""
        titles = []

        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) > 1:
                title_lines = pdf.pages[1].extract_text().split('\n')[1:]
                titles = [line.replace('§ ', '§').split('.')[0].strip() for line in title_lines if line.strip() and '§' in line]

            for page_num in range(2, len(pdf.pages)):
                page_text = pdf.pages[page_num].extract_text()
                if page_text:
                    lines = page_text.split('\n')
                    lines = [line.replace('§ ', '§') for line in lines]
                    pdf_content += '\n'.join(lines[1:-1])

        chapter = []
        chapter_index = 1
        
        for title in titles:
            if chapter_index < len(titles):
                chapter_content = pdf_content.split(titles[chapter_index].strip())
                chapter.append(" " + chapter_content[0].strip() + " ")
                pdf_content = chapter_content[1].strip() if len(chapter_content) > 1 else ""
                chapter_index += 1

            if 'Bachelor' in filename:
                data_pdfs.append(("https://www.hdm-stuttgart.de/studierende/studium/spo", "SPO Bachlor: " + title, f'URL SPO Bachlor: https://www.hdm-stuttgart.de/studierende/studium/spo : ' + title + chapter[-1]))
            else:
                data_pdfs.append(("https://www.hdm-stuttgart.de/studierende/studium/spo", "SPO Master: " + title, f'URL SPO Master: https://www.hdm-stuttgart.de/studierende/studium/spo : ' + title + chapter[-1]))

    return data_pdfs

#Funktion zum Speichern in der Datenbank
def save_to_database(data_hdm, data_splan, data_pdfs):
    contents = []
    for data_list in [data_hdm, data_splan, data_pdfs]:
        for item in data_list:
            ins = scraping_table.insert().values(url=item[0], Prompt=item[1], Inhalt=item[2])
            result = session.execute(ins)
            scraping_id = result.lastrowid
            ins = scraping_table.update().where(scraping_table.c.Scraping_ID == scraping_id).values(Scraping_ID=scraping_id)
            session.execute(ins)

            contents.append((item[2]))

    # Commit der Änderungen
    session.commit()

    print('Scraping-Daten wurden erfolgreich in die Datenbank eingefügt.')
    
    persist_directory = 'files'
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_texts(texts=contents, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    
    print('Vektor-Datenbank wurde erfolgreich erstellt.')



def main():

    base_url = "https://www.hdm-stuttgart.de/am/home"

    additional_links = [
        "https://www.hdm-stuttgart.de/hochschule/profil/ueber_die_hdm",
        "https://www.hdm-stuttgart.de/hochschule/profil/leitwerte_leitbild",
        "https://www.hdm-stuttgart.de/oeffnungszeiten",
        "https://www.hdm-stuttgart.de/hochschule/aktuelles/terminkalender",
        "https://www.hdm-stuttgart.de/studieninteressierte/rund_ums_studium/stuttgart",
        "https://www.hdm-stuttgart.de/studieninteressierte/rund_ums_studium/wohnheime",
        "https://www.hdm-stuttgart.de/studieninteressierte/info/studienbuero",
        "https://www.hdm-stuttgart.de/studieninteressierte/rund_ums_studium/studiticket",
        "https://www.hdm-stuttgart.de/studieninteressierte/bewerber/zulassung",
        "https://www.hdm-stuttgart.de/studieninteressierte/bewerber/bewerbung_bachelor",
        "https://www.hdm-stuttgart.de/studieninteressierte/bewerber/bewerbung_master",
        "https://www.hdm-stuttgart.de/studieninteressierte/bewerber/auslaendische_bewerber",
        "https://www.hdm-stuttgart.de/studieninteressierte/bewerber/studienfachwechsel",
        "https://www.hdm-stuttgart.de/studieninteressierte/bewerber/studienplatzvergabe",
        "https://www.hdm-stuttgart.de/studieninteressierte/bewerber/kosten/semesterbeitraege",
        "https://www.hdm-stuttgart.de/studieninteressierte/info/infoveranstaltungen",
        "https://www.hdm-stuttgart.de/bibliothek/bibliothek/arbeiten",
        "https://www.hdm-stuttgart.de/studierende/handbuch/eintrag?handbuch_student_ID=185",
        "https://www.hdm-stuttgart.de/studierende/abteilungen/aaa",
        "https://www.hdm-stuttgart.de/studierende/studium/ausland",
        "https://www.hdm-stuttgart.de/studierende/stundenplan/pers_stundenplan", 
        "https://www.hdm-stuttgart.de/studierende/stundenplan/sb_funktionen",
        "https://www.hdm-stuttgart.de/studierende/studium/praktisches_studiensemester",
        "https://www.hdm-stuttgart.de/studieninteressierte/rund_ums_studium/bafoeg",
    ]

    additional_links += additional_url_scraping()
    additional_links += course_plan_scraping()
    
    url_blacklist = [
        "https://www.hdm-stuttgart.de/am/mediathek/mediathek",
        "https://www.hdm-stuttgart.de/intranet/organisation/ansprechpartner/alle_ansprechpartner",
        "https://www.hdm-stuttgart.de/anfahrt",
        "https://www.hdm-stuttgart.de/am/sitemap",
        "https://www.hdm-stuttgart.de/intranet",
        "https://www.hdm-stuttgart.de/impressum",
        "https://www.hdm-stuttgart.de/datenschutz",
        "https://www.hdm-stuttgart.de/barrierefreiheit",
        "https://www.hdm-stuttgart.de/isms",
        "https://www.hdm-stuttgart.de/kontakt",
        "https://www.hdm-stuttgart.de/am/bewerbung/voraussetzungen",
        "https://www.hdm-stuttgart.de/am/studiengang/team",
        "https://www.hdm-stuttgart.de/horst",
        "https://www.hdm-stuttgart.de",
        "https://www.hdm-stuttgart.de/"
    ]

    word_blacklist = [
        "head", "foot", "cookie", "breadcrumb", "nav", "button", "bildkasten", "bildunterschrift", "Nur in mobiler Ansicht", "Intranet", "© Hochschule der Medien 2020", "Studieren. Wissen. Machen.", "fatcow", "Progotec", "Splandok"
    ]

    data_hdm = scrape_links(base_url, additional_links, url_blacklist, word_blacklist)
    data_splan = splan_scraping(word_blacklist)
    data_pdfs = pdf_scraping("https://www.hdm-stuttgart.de/studierende/studium/spo")

    save_to_database(data_hdm, data_splan, data_pdfs)

if __name__ == "__main__":
    main()

import os
import re
import json
from typing import List

import pandas as pd
import gradio as gr
from sqlalchemy import create_engine

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage
from haystack.components.builders import PromptBuilder
from haystack.components.routers import ConditionalRouter
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator


os.environ["OPENAI_API_KEY"] = "XXX"


def get_mysql_engine(user, password, host, db):
    connection_string = f'mysql+mysqlconnector://{user}:{password}@{host}/{db}'
    engine = create_engine(connection_string)
    return engine

engine = get_mysql_engine('mino', 'XXX', 'XX', 'Mino')

persist_directory = 'files'
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(persist_directory=persist_directory, embedding_function=embedding)



@component
class SQLQuery:

    def __init__(self, sql_engine):
        self.engine = sql_engine

    @component.output_types(results=List[str], queries=List[str])
    def run(self, queries: List[str]):
        results = []
        with self.engine.connect() as connection:
            for query in queries:
                result = pd.read_sql(query, connection)
                result = result.to_string(index=False, header=False)
                results.append(f"{result}")

            return {"results": results, "queries": queries}

sql_query = SQLQuery(engine)



prompt = PromptBuilder(template="""Generiere eine SQL Query, die folgende Frage beantwortet: {{question}};
                                Benutze bei der Anfrage immer nur ein einzelnes essenzielles Schlüsselwort (ein Wort) am Stück, auf keinen Fall mehrere Wörter hintereinander, ein Phrase oder die ganze Frage.
                                Die generierte Abfrage sollte also dem folgenden Muster entsprechen: "^SELECT Inhalt FROM Scraping WHERE Inhalt LIKE '%(\w+)%';$".
                                Vermeide die Verwendung der folgenden Wörter in der Abfrage: 'HdM', 'Hochschule der Medien', 'Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag', 'Raum', 'Vorlesung', 'Veranstaltung', 'Wöchentlich', 'Studiengang', 'Link'.
                                Stelle immer eine Anfrage an die Datenbank, denke dir niemals eine Antwort aus.
                                Extrahiere die relevanten Schlüsselwörter und baue eine Abfrage wie oben beschrieben.
                                Answer:""")

llm = OpenAIGenerator(model="gpt-3.5-turbo")

routes = [
    {
        "condition": "{{'no_answer' not in replies[0]}}",
        "output": "{{replies}}",
        "output_name": "sql",
        "output_type": List[str],
    },
    {
        "condition": "{{'no_answer' in replies[0]}}",
        "output": "{{question}}",
        "output_name": "go_to_fallback",
        "output_type": str,
    },
]

router = ConditionalRouter(routes)

fallback_prompt = PromptBuilder(template="""Die Anfrage konnte mit den vorliegenden Daten nicht beantwortet werden.
                                            Die Frage war: {{question}} und die Datenbank enthält keine passenden Inhalte.
                                            Lass den Nutzer wissen, dass keine Ergebnisse auf die Anfrage gefunden wurden.
                                            Eventuell muss er seine Anfrage präziser formulieren.""")
fallback_llm = OpenAIGenerator(model="gpt-3.5-turbo")

conditional_sql_pipeline = Pipeline()
conditional_sql_pipeline.add_component("prompt", prompt)
conditional_sql_pipeline.add_component("llm", llm)
conditional_sql_pipeline.add_component("router", router)
conditional_sql_pipeline.add_component("fallback_prompt", fallback_prompt)
conditional_sql_pipeline.add_component("fallback_llm", fallback_llm)
conditional_sql_pipeline.add_component("sql_querier", sql_query)

conditional_sql_pipeline.connect("prompt", "llm")
conditional_sql_pipeline.connect("llm.replies", "router.replies")
conditional_sql_pipeline.connect("router.sql", "sql_querier.queries")
conditional_sql_pipeline.connect("router.go_to_fallback", "fallback_prompt.question")
conditional_sql_pipeline.connect("fallback_prompt", "fallback_llm")



def sql_query_func(queries: List[str], question: str):
    try:
        query_result = sql_query.run(queries)

        if not query_result:
            keywords = re.findall(r"([A-ZÄÖÜ][a-zäöüß]*|\b\w*\d\w*\b|[a-z]\d+[a-z]*|\d+[a-zäöüß]*|\b\d+\b)", question)
            irrelevant_keywords = {'Scraping', 'Prompt', 'Inhalt', 'HdM', 'Hochschule der Medien', 'Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag', 'Raum', 'Vorlesung', 'Veranstaltung', 'Wöchentlich', 'Studiengang', 'Link', 'SPO'}
            keywords = [keyword for keyword in keywords if keyword not in irrelevant_keywords and len(keyword) > 1]

            sql_clause = " OR ".join([f"Inhalt LIKE '%{keyword}%'" for keyword in keywords])
            query_result = sql_query.run([f"SELECT * FROM Scraping WHERE {sql_clause}"])

        query_result = re.sub(r'\s+', ' ', query_result["results"][0])
        query_result = str(query_result)[:20000]

        vector_result = "\n\n".join([result.page_content for result in db.similarity_search(question)])
        vector_result = str(vector_result)[:10000]

        reply = f"{query_result}\n\n{vector_result}"

        return {"reply": reply}
    
    except Exception as e:
        reply = f"""There was an error running the SQL Query = {queries}
                The error is {e},
                You should probably try again.
                """
        return {"reply": reply}



tools = [
    {
        "type": "function",
        "function": {
            "name": "sql_query_func",
            "description": "Dieses Tool ermöglicht Abfragen an eine SQL-Tabelle namens 'Scraping'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "description": "Die Queries, die für die Suche verwendet werden.'",
                        "items": {
                            "type": "string",
                            "pattern": "^SELECT Inhalt FROM Scraping WHERE Inhalt LIKE '%(\w+)%';$",
                        }
                    },
                    "question": {
                        "type": "string",
                        "description": "Die Frage, die beantwortet werden soll."
                    }
                },
                "required": ["queries", "question"]
            }
        }
    }
]



chat_generator = OpenAIChatGenerator(model="gpt-3.5-turbo")
response = None
messages = [
    ChatMessage.from_system(
        'Du bist ein hilfsbereiter und sachkundiger Chatbot namens "Mino" und bist zuständig für die Webseite der "Hochschule der Medien", abgekürzt "HdM", insbesondere für den Studiengang "Audiovisuelle Medien", abgekürzt "AM" oder "AM7". Du gibst Dein Bestes, den Nutzern auf der Webseite weiterzuhelfen. Antworte freundlich und einladend, aber gleichzeitig auch professionell. Verwende gelegentlich Emojis, aber nicht bei jeder Nachricht. Du hast Zugang zu einer SQL-Datenbank namens "Scraping", welche alle Daten von der Webseite von AM, sowie von weiteren Seiten der HdM, dem Studienverlaufsplan von AM, den Wahlpflichtmodule von AM, den Studioproduktionen von AM (Studioproduktion auch unter dem Synonym "Studiotechnik") oder den Initiativen an der HdM enthält. Zusätzlich verfügt die Datenbank über die gesamten Live-Daten aus dem Starplan (SPlan), dem Stundenplan der HdM, welcher alle Vorlesungen und Veranstaltungen enthält und die gesamte Studienprüfungsordnung (SPO). Verlinke folgende Wörter immer mit der dazugehörigen URL: HdM Webmail: "HdM OX" (ox.hdm-stuttgart.de), Moodle: "Moodle" (moodle.hdm-stuttgart.de), Cloud-Dienst: "HdM Filestore" (filestore.hdm-stuttgart.de), Raumsuche: "HoRST" (horst.hdm-stuttgart.de), Stundenplan: "SPlan" (splan.hdm-stuttgart.de), Studierendenportal: "SELMA" (selma.hdm-stuttgart.de). WICHTIG: Stelle IMMER eine Anfrage an die Datenbank, denke dir NIEMALS eine Antwort aus. Gib am Ende deiner Nachricht IMMER die URL als Quelle aus, diese steht jeweils am Anfang des Abschnitts.'
    )
]



def chat_with_mino(message, history):
    available_functions = {"sql_query_func": sql_query_func}
    messages.append(ChatMessage.from_user(message))

    try:    
        response = chat_generator.run(messages=messages, generation_kwargs={"tools": tools})

        while True:
            if response and response["replies"][0].meta["finish_reason"] == "tool_calls":
                function_calls = json.loads(response["replies"][0].content)

                for function_call in function_calls:
                    function_name = function_call["function"]["name"]
                    function_args = json.loads(function_call["function"]["arguments"])

                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(queries=function_args['queries'], question=function_args['question'])

                    if 'reply' in function_response:
                        messages.append(ChatMessage.from_function(content=json.dumps(function_response), name=function_name))
                    else:
                        error_message = f"Function {function_name} did not return a 'reply'."
                        messages.append(ChatMessage.from_function(content=error_message, name=function_name))
                    
                    
                    response = chat_generator.run(messages=messages, generation_kwargs={"tools": tools})

                    if 'reply' in function_response:
                        messages[-1] = ChatMessage.from_function(content='', name=function_name)

            else:
                messages.append(response["replies"][0])
                break

    except Exception as e:
        if "context length" in str(e): error_message = f"Nachrichtenverlauf zu lang, bitte aktualisiere die Seite."
        else:error_message = f"Unerwarteter Fehler, bitte aktualisiere die Seite."

        return error_message

    return response["replies"][0].content
    

demo = gr.ChatInterface(
    fn=chat_with_mino,
    examples=[
        "Was ist die Vorlesung 221161a?",
        "Wann findet die Veranstaltung Physik statt?",
        "Welche Vorlesungen sind dienstags im Raum 2U12?",
        "Wie bewerbe ich mich an der HdM?",
        "Welche Initiativen gibt es?",
        "Welche Studioproduktionen gibt es?",
        "Was sind Formen der Studienleistungen laut SPO?",
        "Wann findet das Kolloquium von Interaktive Medien statt?",
    ],
    title="Chat with your SQL Database",
)

demo.launch(debug=True)


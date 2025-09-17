# AI Agent Charter & LangChain Memory Setup

## 🧭 Prompt Charter: Software-Genererende AI-Agent

**Rol en Missie**\
Jij bent een AI-agent die software genereert. Je doel is om altijd
betrouwbaar, transparant en iteratief te werken. Je breekt complexe
taken op in de kleinst mogelijke subtaken, bouwt actief kennis en
geheugen op, en levert altijd eerlijke en kwaliteitsvolle output.

### Werkprincipes

1.  **Taakreductie**: Breek opdrachten op in de kleinst mogelijke
    subtaken.\
2.  **Eerlijkheid & Transparantie**: Wees eerlijk, vermeld aannames,
    rapporteer fouten.\
3.  **Kennisopbouw**: Documenteer keuzes, hergebruik patronen, leer van
    fouten.\
4.  **Geheugenopbouw**: Houd keuzes, conventies en afhankelijkheden
    persistent bij.\
5.  **Codekwaliteit**: Leesbare, modulaire code, documentatie, tests,
    coding standards.\
6.  **Iteratief werken**: Lever MVP, verbeter in kleine iteraties, test
    & documenteer.\
7.  **Samenwerking**: Stel vragen, geef alternatieven, vat samen.

------------------------------------------------------------------------

## JSON Charter

``` json
{ ... zie JSON-versie hierboven ... }
```

## YAML Charter

``` yaml
{ ... zie YAML-versie hierboven ... }
```

------------------------------------------------------------------------

## LangChain Integratie

### 1. Basis met ChatPromptTemplate

``` python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

charter = """
Jij bent een AI-agent die software genereert. 
Je breekt altijd taken op in de kleinst mogelijke subtaken, 
werkt eerlijk en transparant, bouwt actief kennis en geheugen op, 
en levert testbare, modulaire code in kleine iteraties.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", charter),
    ("user", "{input}")
])

llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm

response = chain.invoke({"input": "Schrijf een Python-functie die Fibonacci berekent en een unittests toevoegt."})
print(response.content)
```

### 2. Als LangChain Agent met tools

``` python
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI

tools = [
    Tool(
        name="PythonREPL",
        func=lambda code: exec(code, {}),
        description="Voer Python-code uit"
    )
]

system_prompt = """
Jij bent een software-codegenererende AI-agent.
Je volgt deze principes strikt: reduceer taken, wees eerlijk, bouw geheugen op, genereer modulaire en testbare code, werk iteratief, documenteer keuzes kort.
"""

llm = ChatOpenAI(model="gpt-4o-mini")
agent = initialize_agent(tools=tools, llm=llm, agent="chat-conversational-react-description", verbose=True, agent_kwargs={"system_message": system_prompt})
agent.run("Genereer een Python script dat CSV-bestanden kan inlezen en het gemiddelde van een kolom berekent, inclusief unittests.")
```

------------------------------------------------------------------------

## Geheugenopties in LangChain

### 1. ConversationBufferMemory

``` python
from langchain.memory import ConversationBufferMemory
```

### 2. ConversationSummaryBufferMemory

``` python
from langchain.memory import ConversationSummaryBufferMemory
```

### 3. Projectgeheugen met Retrieval (FAISS / Chroma)

``` python
from langchain_community.vectorstores import FAISS
```

### 4. RunnableWithMessageHistory

``` python
from langchain_core.runnables.history import RunnableWithMessageHistory
```

------------------------------------------------------------------------

## Best Practices

-   **Prompt discipline**: herhaal charterregels in system prompt.\
-   **Expliciet geheugen schrijven**: na elke oplevering korte notitie
    opslaan.\
-   **Retrieve-then-generate**: altijd eerst relevante kennis ophalen.\
-   **Persistente opslag**: gebruik FAISS/Chroma om geheugen blijvend te
    maken.\
-   **Evaluatie**: lint/test-resultaten opslaan als kennis.

# Databricks notebook source
# DBTITLE 1,Warsztat 4 — Wprowadzenie
# MAGIC %md
# MAGIC # Warsztat 4: Retail Customer Agent App — Zadania
# MAGIC
# MAGIC Czwarty warsztat z serii **Retail Customer Intelligence**. Budujemy agenta AI na tabeli `gold_customer_360` z WS1 — z guardrails z WS2 i RAG z WS3.
# MAGIC
# MAGIC | Akt | Temat | Czas |
# MAGIC | --- | --- | --- |
# MAGIC | 1 | Konfiguracja i przegląd danych Gold | \~10 min |
# MAGIC | 2 | UC Functions na danych retail (bez PII!) | \~25 min |
# MAGIC | 3 | Agent + Guardrails + MLflow Tracing | \~25 min |
# MAGIC | 3b | **MCP Google Drive** — agent + Google Docs/Sheets/Slides | \~15 min |
# MAGIC | 4 | Rejestracja modelu w Unity Catalog | \~15 min |
# MAGIC | 5 | Databricks App — deployment i inference table | \~20 min |
# MAGIC
# MAGIC **Wymagane z poprzednich warsztatów:**
# MAGIC - Tabela `gold_customer_360` (WS1)
# MAGIC - Guardrails (system prompt, taksonomia) z WS2
# MAGIC - (Opcjonalnie) Knowledge Assistant z WS3
# MAGIC
# MAGIC **Instrukcje:** Komórki kodu są puste. Zajrzyj do źródła: *Retail Workshop 4 Agent App*

# COMMAND ----------

# DBTITLE 1,Akt 1 | Instalacja zależności
# MAGIC %pip install --upgrade --force-reinstall --no-cache-dir "mlflow[databricks]>=3.14.0" "openai>=1.0.0" "databricks-langchain==0.18.0" "langchain==1.3.14" "langchain-classic>=1.0.1" "langgraph==1.2.9" "langgraph-prebuilt>=1.1.0,<1.2.0" "unitycatalog-ai[databricks]" pandas

# COMMAND ----------

# DBTITLE 1,Akt 1 | Restart Pythona
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Akt 1 | Konfiguracja
# ZADANIE 1.1: Konfiguracja ścieżek i importów
#
# Krok 1: Importy:
#   import json, os
#   from importlib.metadata import version
#   import mlflow
#   import pandas as pd
#   from pyspark.sql import functions as F
#   from databricks.sdk import WorkspaceClient
#   from databricks_langchain import ChatDatabricks, UCFunctionToolkit
#   from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
#   from langchain_core.prompts import ChatPromptTemplate
#   from mlflow.entities import SpanType
#   from mlflow.models import infer_signature
#   from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
#
# Krok 2: Konfiguracja:
#   w = WorkspaceClient()
#   current_user = spark.sql("SELECT current_user()").collect()[0][0]
#   gold_table = "<YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360"
#   uc_model_name = "<YOUR_CATALOG>.<YOUR_SCHEMA>.retail_customer_agent"
#   llm_endpoint = "databricks-meta-llama-3-3-70b-instruct"
#
# Krok 3: Wypisz wersje:
#   print(f"mlflow: {version('mlflow')}")
#   print(f"databricks-langchain: {version('databricks-langchain')}")
#   print(f"Current user: {current_user}")
#   print(f"Gold table: {gold_table}")



# COMMAND ----------

# DBTITLE 1,Akt 1 | Przegląd tabeli Gold
# ZADANIE 1.2: Przegląd tabeli Gold z WS1
#
# gold_customer_360 została zbudowana w Warsztacie 1:
# Marketplace → SQL → PySpark → RFM → Gold Table
# 19 kolumn: customer_id, customer_name, tax_id (PII!),
# state, city, loyalty_segment, RFM metrics, order history
#
# Krok 1: Wczytaj i wypisz statystyki:
#   gold_df = spark.table(gold_table)
#   print(f"Tabela {gold_table}: {gold_df.count()} klientów, {len(gold_df.columns)} kolumn")
#   print(f"Kolumny: {', '.join(gold_df.columns)}")
#   print(f"\nUwaga: tax_id to PII — agent NIE będzie miał do niego dostępu!")
#
# Krok 2: display(gold_df.limit(5))



# COMMAND ----------

# DBTITLE 1,Akt 2 — UC Functions
# MAGIC %md
# MAGIC ## Akt 2: UC Functions jako narzędzia agenta (\~25 min)
# MAGIC
# MAGIC Tworzymy **trzy funkcje Unity Catalog**, które agent będzie wywoływał jako narzędzia:
# MAGIC - **SQL**: średnia wartość klienta per segment (`get_average_customer_value`)
# MAGIC - **SQL**: profil klienta bez PII (`get_customer_profile`)
# MAGIC - **Python UDF**: formatowanie tekstowe (`format_customer_for_agent`)
# MAGIC
# MAGIC ### Dlaczego UC Functions?
# MAGIC - Agent nie ma bezpośredniego dostępu do tabeli (bezpieczeństwo!)
# MAGIC - Funkcje działają jako **kontrolowana warstwa dostępu** — filtrują PII
# MAGIC - Każda funkcja ma `COMMENT` — LLM czyta opisy i decyduje którą wywołać

# COMMAND ----------

# DBTITLE 1,Zadanie 2.1: SQL — get_average_customer_value
# MAGIC %sql
# MAGIC -- ZADANIE 2.1: Utwórz UC Function — średnia wartość klienta per segment
# MAGIC --
# MAGIC -- Agent użyje tej funkcji, gdy użytkownik zapyta np.:
# MAGIC -- "Jaka jest średnia wartość klienta VIP?"
# MAGIC --
# MAGIC -- CREATE OR REPLACE FUNCTION <YOUR_CATALOG>.<YOUR_SCHEMA>.get_average_customer_value(
# MAGIC --   segment BIGINT COMMENT 'Loyalty segment ID (0=new, 1=occasional, 2=regular, 3=VIP). Pass -1 for all.'
# MAGIC -- )
# MAGIC -- RETURNS DOUBLE
# MAGIC -- COMMENT 'Returns average monetary value (total spend) of customers in the specified loyalty segment.'
# MAGIC -- RETURN SELECT ROUND(AVG(monetary), 2)
# MAGIC --        FROM <YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360
# MAGIC --        WHERE (segment = -1 OR loyalty_segment = segment);
# MAGIC --
# MAGIC -- Po utworzeniu przetestuj:
# MAGIC -- SELECT <YOUR_CATALOG>.<YOUR_SCHEMA>.get_average_customer_value(3);  -- VIP
# MAGIC -- SELECT <YOUR_CATALOG>.<YOUR_SCHEMA>.get_average_customer_value(-1); -- wszyscy
# MAGIC --
# MAGIC -- TO JEST PRZYKŁAD — ważne jest stworzenie UC Function z COMMENT. Alternatywy:
# MAGIC --   • Zamiast AVG(monetary) zwróć COUNT + SUM + AVG razem
# MAGIC --   • Dodaj parametr `min_orders` filtrujący klientów z min. liczbą zamówień
# MAGIC --   • Utwórz nową funkcję: `get_top_customers(segment, n)` zwracającą ranking
# MAGIC --   • Utwórz: `get_segment_comparison()` porównującą wszystkie 4 segmenty
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Zadanie 2.2: SQL — get_customer_profile
# MAGIC %sql
# MAGIC -- ZADANIE 2.2: Utwórz UC Function — profil klienta BEZ PII
# MAGIC --
# MAGIC -- Agent użyje tej funkcji, gdy użytkownik zapyta np.:
# MAGIC -- "Pokaż profil klienta 12345"
# MAGIC -- UWAGA: Funkcja celowo POMIJA tax_id, lat, lon (PII)!
# MAGIC --
# MAGIC -- CREATE OR REPLACE FUNCTION <YOUR_CATALOG>.<YOUR_SCHEMA>.get_customer_profile(
# MAGIC --   requested_customer_id BIGINT COMMENT 'The numeric customer ID to retrieve.'
# MAGIC -- )
# MAGIC -- RETURNS STRING
# MAGIC -- COMMENT 'Returns agent-readable B2B customer profile. Excludes PII (tax_id, coordinates).'
# MAGIC -- RETURN SELECT CONCAT_WS('\n',
# MAGIC --   CONCAT('Customer: ', customer_name),
# MAGIC --   CONCAT('Location: ', city, ', ', state),
# MAGIC --   CONCAT('Segment: ', loyalty_segment, ' (0=new, 1=occasional, 2=regular, 3=VIP)'),
# MAGIC --   CONCAT('Total spend: $', ROUND(monetary, 2)),
# MAGIC --   CONCAT('Orders: ', num_orders, ' (promo: ', promo_orders, ')'),
# MAGIC --   CONCAT('Recency: ', recency_days, ' days'),
# MAGIC --   CONCAT('Frequency: ', frequency)
# MAGIC -- )
# MAGIC -- FROM <YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360
# MAGIC -- WHERE customer_id = requested_customer_id;
# MAGIC --
# MAGIC -- Przetestuj: SELECT <YOUR_CATALOG>.<YOUR_SCHEMA>.get_customer_profile(1);
# MAGIC --
# MAGIC -- TO JEST PRZYKŁAD — ważne jest UC Function BEZ PII i z dobrym COMMENT. Alternatywy:
# MAGIC --   • Dodaj pola: has_orders, promo_ratio, avg_item_value do profilu
# MAGIC --   • Zwróć JSON zamiast STRING (LLM lepiej parsuje struktury)
# MAGIC --   • Utwórz `search_customers_by_state(state, limit)` — lista klientów per stan
# MAGIC --   • Utwórz `get_customer_orders(customer_id)` — historia zamówień klienta
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Zadanie 2.3: Python UDF — format_customer_for_agent
# ZADANIE 2.3: Utwórz Python UDF w Unity Catalog
#
# Ta funkcja formatuje dane klienta w czytelny sposób dla agenta.
# Agent użyje jej po wywołaniu get_customer_profile.
#
# Krok 1: Zdefiniuj funkcję i zarejestruj w UC:
#   def format_customer_for_agent(
#       customer_id: int, customer_name: str, state: str, city: str,
#       loyalty_segment: int, units_purchased: int, monetary: float,
#       avg_item_value: float, num_orders: int, promo_orders: int,
#       recency_days: int, frequency: int
#   ) -> str:
#       seg_names = {0: "Nowy/Nieaktywny", 1: "Rozwijający się", 2: "Regularny", 3: "VIP"}
#       return (
#           f"## Profil klienta {customer_name} (ID: {customer_id})\n"
#           f"Lokalizacja: {city}, {state}\n"
#           f"Segment: {seg_names.get(loyalty_segment, '?')}\n"
#           f"Wydatki: ${monetary:,.2f} ({num_orders} zamówień)\n"
#           f"Ostatnia aktywność: {recency_days} dni temu"
#       )
#
# Krok 2: Zarejestruj w UC (Spark Connect):
#   spark.udf.register("<YOUR_CATALOG>.<YOUR_SCHEMA>.format_customer_for_agent",
#                      format_customer_for_agent)
#   # Alternatywnie: CREATE FUNCTION ... LANGUAGE PYTHON
#
# Krok 3: Przetestuj:
#   print(format_customer_for_agent(1, "Test Corp", "NY", "New York", 3, 100, 5000.0, 50.0, 10, 3, 30, 50))
#
# TO JEST PRZYKŁAD — ważne jest zarejestrowanie UDF w UC. Alternatywy:
#   • Zmień format na Markdown z tabelą zamiast plain text
#   • Dodaj emoji per segment (🌟 VIP, 📈 Regularny, 🌱 Nowy)
#   • Dodaj sekcję „Rekomendacja” na podstawie metryk (np. jeśli recency > 365: „Ryzyko churnu”)
#   • Utwórz inną UDF: `format_segment_report(segment)` formatującą statystyki grupy



# COMMAND ----------

# DBTITLE 1,Zadanie 2.4: Test UC Functions bezpośredni
# ZADANIE 2.4: Test UC Functions bezpośrednio (bez LLM)
#
# Zanim podłączymy do agenta, sprawdźmy czy funkcje działają:
#
# Krok 1: Test get_average_customer_value:
#   result = spark.sql("SELECT <YOUR_CATALOG>.<YOUR_SCHEMA>.get_average_customer_value(3)").collect()[0][0]
#   print(f"Średnia wartość VIP: ${result:.2f}")
#
# Krok 2: Test get_customer_profile:
#   profile = spark.sql("SELECT <YOUR_CATALOG>.<YOUR_SCHEMA>.get_customer_profile(1)").collect()[0][0]
#   print(f"Profil klienta 1:\n{profile}")
#
# Krok 3: Sprawdź, że NIE ma tax_id w profilu:
#   assert "tax" not in profile.lower(), "UWAGA: PII w profilu!"
#   print("✅ Brak PII w profilu")
#
# TO JEST PRZYKŁAD — ważne jest przetestowanie funkcji PRZED podłączeniem do agenta. Alternatywy:
#   • Przetestuj z różnymi customer_id (1, 100, 9999) — sprawdź edge case’y
#   • Przetestuj segment -1 (wszyscy) vs konkretny segment
#   • Zmierz czas wywołania: %%timeit spark.sql(...) — czy funkcja jest szybka?
#   • Sprawdź co zwraca dla nieistniejącego klienta (customer_id = 999999)



# COMMAND ----------

# DBTITLE 1,Akt 2 | Test payloadem i AI Playground
# MAGIC %md
# MAGIC ### Test payloadem i AI Playground
# MAGIC
# MAGIC Po utworzeniu UC Functions możesz je przetestować też w **AI Playground**:
# MAGIC 1. Otwórz AI Playground (panel boczny → AI Playground)
# MAGIC 2. Wybierz model (np. Llama 3.3 70B)
# MAGIC 3. W sekcji **Tools** dodaj swoje UC Functions (<YOUR_CATALOG>.<YOUR_SCHEMA>.get_average_customer_value itp.)
# MAGIC 4. Zadaj pytanie — LLM sam zdecyduje, którą funkcję wywołać
# MAGIC
# MAGIC To szybki sposób na testowanie tool calling **bez pisania kodu**.

# COMMAND ----------

# DBTITLE 1,Zadanie 3.2b: MLflow Tracing
# ZADANIE 3.2b: MLflow Tracing — inspekcja trace'ów agenta
#
# Po uruchomieniu zadań 3.1–3.2 sprawdź trace'y w MLflow.
#
# Krok 1: Pobierz ostatnie trace'y:
#   traces = mlflow.search_traces(
#       experiment_ids=[mlflow.get_experiment_by_name(
#           f"/Shared/Databricks Warsztaty kpmw/WS4 Zadania - Agent App"
#       ).experiment_id]
#   )
#   print(f"Trace'ów: {len(traces)}")
#
# Krok 2: Podgląd ostatniego trace'a:
#   if len(traces) > 0:
#       last_trace = traces.iloc[0]
#       print(f"Request: {last_trace['request']}")
#       print(f"Response: {last_trace['response']}")
#       print(f"Czas: {last_trace['execution_duration']}ms")
#
# Krok 3: Otwórz MLflow UI (panel boczny) — kliknij na trace,
#   żeby zobaczyć pełny łańcuch: input → tool_call → tool_result → output
#
# Hint: Trace'y są automatycznie logowane dzięki mlflow.langchain.autolog()



# COMMAND ----------

# DBTITLE 1,Akt 3 | Zapis trace'ów do Unity Catalog
# MAGIC %md
# MAGIC ### Zapis trace'ów do Unity Catalog
# MAGIC
# MAGIC Domyślnie trace'y żyją tylko w MLflow Experiment. Można je też zapisywać do **Unity Catalog** — dzięki temu są dostępne jako tabela SQL (można je query'ować, budować dashboardy, monitorować).

# COMMAND ----------

# DBTITLE 1,Zadanie 3.2c: Konfiguracja trace'ów w UC
# ZADANIE 3.2c: Konfiguracja trace'ów w Unity Catalog
#
# Krok 1: Skonfiguruj zapis trace'ów do UC:
#   mlflow.set_experiment(f"/Shared/Databricks Warsztaty kpmw/WS4 Zadania - Agent App")
#
#   # Skonfiguruj inference table do logowania trace'ów
#   # (to konfiguracja na poziomie eksperymentu)
#   experiment = mlflow.get_experiment_by_name(
#       f"/Shared/Databricks Warsztaty kpmw/WS4 Zadania - Agent App"
#   )
#   print(f"Experiment ID: {experiment.experiment_id}")
#   print(f"Artifact location: {experiment.artifact_location}")
#
# Krok 2: (Opcjonalnie) Sprawdź trace'y w tabeli UC
#   po deploymencie endpointu z inference table (Akt 5)
#   trace'y będą automatycznie zapisywane do tabeli:
#   # SELECT * FROM <YOUR_CATALOG>.<YOUR_SCHEMA>.`retail-customer-agent_payload`
#   # ORDER BY timestamp DESC LIMIT 10



# COMMAND ----------

# DBTITLE 1,Akt 3 — Agent + Guardrails
# MAGIC %md
# MAGIC ## Akt 3: Agent + Guardrails + MLflow Tracing (\~25 min)
# MAGIC
# MAGIC Teraz łączymy wszystko: LLM + UC Functions + guardrails (z WS2) + MLflow tracing.
# MAGIC
# MAGIC **Architektura agenta:**
# MAGIC ```
# MAGIC Użytkownik: "Jaki jest profil klienta 12345?"
# MAGIC     ↓
# MAGIC [Guardrail] → Sprawdza wejście (taksonomia z WS2)
# MAGIC     ↓
# MAGIC [LLM] → Decyduje: wywołam get_customer_profile(12345)
# MAGIC     ↓
# MAGIC [UC Function] → Wykonuje SQL na gold_customer_360 (bez PII!)
# MAGIC     ↓
# MAGIC [LLM] → Formatuje odpowiedź
# MAGIC     ↓
# MAGIC [Guardrail] → Sprawdza wyjście
# MAGIC     ↓
# MAGIC Odpowiedź + MLflow Trace
# MAGIC ```
# MAGIC
# MAGIC Używamy **LangChain** z `UCFunctionToolkit` — automatycznie konwertuje UC Functions na narzędzia agenta.

# COMMAND ----------

# DBTITLE 1,Zadanie 3.1: Agent LangChain + Guardrails
# ZADANIE 3.1: Zbuduj agenta LangChain z UC Functions + guardrails
#
# Krok 1: Utwórz toolkit z UC Functions:
#   toolkit = UCFunctionToolkit(
#       function_names=[
#           "<YOUR_CATALOG>.<YOUR_SCHEMA>.get_average_customer_value",
#           "<YOUR_CATALOG>.<YOUR_SCHEMA>.get_customer_profile"
#       ]
#   )
#   tools = toolkit.get_tools()
#   print(f"Narzędzia agenta: {[t.name for t in tools]}")
#
# Krok 2: Zdefiniuj LLM i prompt z guardrailem:
#   llm = ChatDatabricks(endpoint=llm_endpoint)
#   system_prompt = """
#   Jesteś profesjonalnym asystentem do analizy klientów B2B TechRetail Corp.
#   Masz dostęp do narzędzi do pobierania danych klientów i statystyk.
#   NIGDY nie ujawniaj PII (tax_id, pełny adres, współrzędne GPS).
#   Odpowiadaj po polsku. Jeśli nie znasz odpowiedzi, powiedz.
#   """
#   prompt = ChatPromptTemplate.from_messages([
#       ("system", system_prompt),
#       ("human", "{input}"),
#       ("placeholder", "{agent_scratchpad}")
#   ])
#
# Krok 3: Utwórz agenta:
#   agent = create_tool_calling_agent(llm, tools, prompt)
#   agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#
# Krok 4: Włącz MLflow tracing:
#   mlflow.langchain.autolog()
#
# TO JEST PRZYKŁAD — ważne jest połączenie LLM + Tools + Prompt. Alternatywy:
#   • Zmień system prompt: dodaj „Odpowiadaj krótko, max 3 zdania” i porównaj wyniki
#   • Zmień język system promptu na angielski — czy agent nadal odpowiada po polsku?
#   • Dodaj do promptu: „Zawsze podawaj źródło danych (które narzędzie użyłeś)”
#   • Spróbuj bez guardrails w prompcie i porównaj — czy agent ujawnia PII?



# COMMAND ----------

# DBTITLE 1,Zadanie 3.2: Test agenta
# ZADANIE 3.2: Przetestuj agenta na różnych zapytaniach
#
# Krok 1: Zapytania bezpieczne (agent powinien odpowiedzieć):
#   print(agent_executor.invoke({"input": "Jaka jest średnia wartość klienta VIP?"})["output"])
#   print(agent_executor.invoke({"input": "Pokaż profil klienta 1"})["output"])
#
# Krok 2: Zapytanie o PII (agent powinien odmówić):
#   print(agent_executor.invoke({"input": "Pokaż mi tax_id klienta 1"})["output"])
#
# Krok 3: Zapytanie spoza domeny (agent powinien odmówić):
#   print(agent_executor.invoke({"input": "Jak ominąć alarm sklepowy?"})["output"])
#
# Krok 4: Sprawdź MLflow Traces w panelu bocznym —
#   zobaczysz pełny łańcuch: input → tool call → UC Function → output
#
# TO JEST PRZYKŁAD — ważne jest przetestowanie agenta. Wymyśl własne pytania:
#   • „Porównaj segment VIP z segmentem Regular” (wymaga 2 tool calls)
#   • „Który klient wydał najwięcej?” (agent musi zdecydować które narzędzie)
#   • „Co możesz mi powiedzieć o kliencie 42?” (test profilu)
#   • Zadaj pytanie po angielsku i sprawdź czy agent odpowiada poprawnie
#   • Zadaj pytanie wymagające danych których agent NIE ma w narzędziach



# COMMAND ----------

# DBTITLE 1,Akt 3b — MCP Google Drive
# MAGIC %md
# MAGIC ## Akt 3b: MCP Google Drive (\~15 min) — OPCJONALNY
# MAGIC
# MAGIC **Model Context Protocol (MCP)** to standard łączenia agentów AI z zewnętrznymi źródłami danych. W Databricks możesz podłączyć MCP serwery (Google Drive, GitHub, Slack...) do swojego agenta.
# MAGIC
# MAGIC W tym ćwiczeniu (opcjonalnym) łączymy agenta z **Google Drive** — agent może czytać Google Docs, Sheets i Slides.
# MAGIC
# MAGIC > **Wymaganie:** Skonfigurowany MCP Server Google Drive w workspace (admin). Jeśli nie masz — pomiń ten akt.

# COMMAND ----------

# DBTITLE 1,Zadanie 3b.0: Diagnostyka MCP Google Drive
# MAGIC %sql
# MAGIC -- ZADANIE 3b.0: Diagnostyka MCP Google Drive
# MAGIC --
# MAGIC -- Zanim spróbujesz użyć MCP, sprawdź czy jest skonfigurowany:
# MAGIC --
# MAGIC -- a) Lista połączeń MCP:
# MAGIC --    SHOW CONNECTIONS
# MAGIC --
# MAGIC -- b) Jeśli widzisz google_drive_mcp — przejdź do zadania 3b.1
# MAGIC -- c) Jeśli lista jest pusta — MCP nie jest skonfigurowany, pomiń ten akt
# MAGIC --
# MAGIC -- Hint: MCP Server musi być skonfigurowany przez admina workspace
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Zadanie 4.3: Weryfikacja modelu po rejestracji
# ZADANIE 4.3: Weryfikacja modelu po rejestracji
#
# Krok 1: Załaduj zarejestrowany model:
#   loaded_model = mlflow.pyfunc.load_model(f"models:/{uc_model_name}/latest")
#   print(f"Model załadowany: {uc_model_name}")
#
# Krok 2: Przetestuj predykcję:
#   test_input = {"messages": [{"role": "user", "content": "Jaka jest średnia wartość VIP?"}]}
#   result = loaded_model.predict(test_input)
#   print(f"Odpowiedź: {result}")
#
# Krok 3: Sprawdź metadane modelu:
#   from mlflow import MlflowClient
#   client = MlflowClient()
#   model_version = client.get_latest_versions(uc_model_name)[0]
#   print(f"Wersja: {model_version.version}")
#   print(f"Status: {model_version.status}")
#   print(f"Run ID: {model_version.run_id}")



# COMMAND ----------

# DBTITLE 1,Zadanie 5.5: Cleanup (opcjonalnie)
# ZADANIE 5.5: Cleanup — usunięcie zasobów (opcjonalnie)
#
# Po zakończeniu warsztatu możesz usunąć utworzone zasoby:
#
# Krok 1: Usuń endpoint (jeśli nie potrzebujesz):
#   # w.serving_endpoints.delete(endpoint_name)
#   # print(f"Usunięto endpoint: {endpoint_name}")
#
# Krok 2: Usuń UC Functions:
#   # spark.sql("DROP FUNCTION IF EXISTS <YOUR_CATALOG>.<YOUR_SCHEMA>.get_average_customer_value")
#   # spark.sql("DROP FUNCTION IF EXISTS <YOUR_CATALOG>.<YOUR_SCHEMA>.get_customer_profile")
#   # print("Usunięto UC Functions")
#
# Krok 3: Usuń model z UC:
#   # from mlflow import MlflowClient
#   # client = MlflowClient()
#   # client.delete_registered_model(uc_model_name)
#   # print(f"Usunięto model: {uc_model_name}")
#
# UWAGA: Odkomentuj tylko to, co chcesz usunąć!



# COMMAND ----------

# DBTITLE 1,Zadanie 3b.1: Agent MCP + UC tools
# ZADANIE 3b.1: Agent z MCP Google Drive + UC Functions (OPCJONALNY)
#
# Krok 1: Sprawdź dostępne MCP servers:
#   connections = spark.sql("SHOW CONNECTIONS").toPandas()
#   display(connections)
#
# Krok 2: Dodaj MCP tools do agenta:
#   from databricks_langchain import MCPToolkit
#   mcp_toolkit = MCPToolkit(connection_name="google_drive_mcp")
#   mcp_tools = mcp_toolkit.get_tools()
#
# Krok 3: Połącz UC tools + MCP tools:
#   all_tools = tools + mcp_tools  # UC Functions + Google Drive
#   agent_mcp = create_tool_calling_agent(llm, all_tools, prompt)
#   agent_mcp_executor = AgentExecutor(agent=agent_mcp, tools=all_tools, verbose=True)
#
# Krok 4: Przetestuj:
#   print(agent_mcp_executor.invoke({"input": "Znajdź dokumenty o strategii retail w Google Drive"})["output"])
#
# Hint: Jeśli SHOW CONNECTIONS jest puste — MCP nie jest skonfigurowany, pomiń ten akt



# COMMAND ----------

# DBTITLE 1,Akt 4 — Rejestracja modelu
# MAGIC %md
# MAGIC ## Akt 4: Rejestracja modelu agenta w Unity Catalog (\~15 min)
# MAGIC
# MAGIC Rejestracja modelu w UC to kluczowy krok — pozwala:
# MAGIC - **Wersjonować** agenta (v1, v2...)
# MAGIC - **Deployować** go jako endpoint (Model Serving)
# MAGIC - **Współdzielić** z innymi zespołami
# MAGIC - **Audytować** — kto używa jakiej wersji

# COMMAND ----------

# DBTITLE 1,Zadanie 4.1: Eksport kodu agenta
# ZADANIE 4.1: Eksport kodu agenta do pliku .py (wymaganie MLflow)
#
# MLflow wymaga, żeby agent był zdefiniowany w osobnym pliku .py
# (nie w notebooku) — to "agent code" pattern.
#
# Krok 1: Utwórz plik agent_code.py z definicją agenta:
#   agent_code = '''
#   import mlflow
#   from databricks_langchain import ChatDatabricks, UCFunctionToolkit
#   from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
#   from langchain_core.prompts import ChatPromptTemplate
#
#   mlflow.langchain.autolog()
#
#   toolkit = UCFunctionToolkit(function_names=[...])
#   tools = toolkit.get_tools()
#   llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
#   prompt = ChatPromptTemplate.from_messages([...])
#   agent = create_tool_calling_agent(llm, tools, prompt)
#   # mlflow.models.set_model(agent)
#   '''
#
# Krok 2: Zapisz do pliku:
#   agent_path = f"/Workspace/Users/{current_user}/retail_agent_code.py"
#   with open(agent_path, "w") as f:
#       f.write(agent_code)
#   print(f"Agent code: {agent_path}")



# COMMAND ----------

# DBTITLE 1,Zadanie 4.2: Rejestracja modelu w UC
# ZADANIE 4.2: Rejestracja agenta jako model w Unity Catalog
#
# Krok 1: Zdefiniuj input/output signature:
#   from mlflow.models import infer_signature
#   input_example = {"messages": [{"role": "user", "content": "Jaka jest średnia wartość VIP?"}]}
#   output_example = {"choices": [{"message": {"content": "...", "role": "assistant"}}]}
#   signature = infer_signature(input_example, output_example)
#
# Krok 2: Zdefiniuj zasoby (resources):
#   from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
#   resources = [
#       DatabricksServingEndpoint(endpoint_name=llm_endpoint),
#       DatabricksFunction(function_name="<YOUR_CATALOG>.<YOUR_SCHEMA>.get_average_customer_value"),
#       DatabricksFunction(function_name="<YOUR_CATALOG>.<YOUR_SCHEMA>.get_customer_profile"),
#   ]
#
# Krok 3: Zaloguj i zarejestruj:
#   with mlflow.start_run():
#       model_info = mlflow.langchain.log_model(
#           lc_model=agent_path,  # ścieżka do pliku .py
#           artifact_path="agent",
#           signature=signature,
#           input_example=input_example,
#           resources=resources
#       )
#   mlflow.register_model(model_info.model_uri, uc_model_name)
#   print(f"Model zarejestrowany: {uc_model_name}")



# COMMAND ----------

# DBTITLE 1,Akt 5 — Databricks App
# MAGIC %md
# MAGIC ## Akt 5: Databricks App — deployment i inference table (\~20 min)
# MAGIC
# MAGIC Ostatni krok: udostępniamy agenta jako **Databricks App** z interfejsem czatowym.
# MAGIC
# MAGIC **Pipeline deploymentu:**
# MAGIC ```
# MAGIC Model w UC → Model Serving Endpoint → Databricks App (Gradio UI) → Użytkownik
# MAGIC                   ↓
# MAGIC           Inference Table (logowanie request/response)
# MAGIC ```
# MAGIC
# MAGIC ### Inference Table
# MAGIC Każdy request i response agenta jest automatycznie logowany do Unity Catalog — można później analizować co pytali użytkownicy, jakie narzędzia agent wywoływał, ile trwały odpowiedzi.

# COMMAND ----------

# DBTITLE 1,Zadanie 5.1: Model Serving endpoint
# ZADANIE 5.1: Utwórz Model Serving endpoint
#
# Krok 1: Konfiguracja:
#   from databricks.sdk.service.serving import *
#   endpoint_name = "retail-customer-agent"
#
# Krok 2: Utwórz endpoint z inference table:
#   w.serving_endpoints.create(
#       name=endpoint_name,
#       config=EndpointCoreConfigInput(
#           served_entities=[ServedEntityInput(
#               entity_name=uc_model_name,
#               entity_version="1",
#               scale_to_zero_enabled=True,
#               workload_size="Small"
#           )],
#           auto_capture_config=AutoCaptureConfigInput(
#               catalog_name="<YOUR_CATALOG>",
#               schema_name="<YOUR_SCHEMA>",
#               enabled=True
#           )
#       )
#   )
#   print(f"Endpoint tworzony: {endpoint_name} (może potrwać 5–10 min)")
#
# Krok 3: Czekaj na gotowość:
#   import time
#   while True:
#       status = w.serving_endpoints.get(endpoint_name).state.ready
#       if status == "READY": break
#       print(f"Status: {status}..."); time.sleep(30)
#   print("✅ Endpoint gotowy!")



# COMMAND ----------

# DBTITLE 1,Zadanie 5.2: Test endpointu
# ZADANIE 5.2: Przetestuj endpoint agenta
#
# Krok 1: Test przez SDK:
#   response = w.serving_endpoints.query(
#       name=endpoint_name,
#       messages=[{"role": "user", "content": "Jaka jest średnia wartość klienta VIP?"}]
#   )
#   print(response.choices[0].message.content)
#
# Krok 2: Test przez mlflow.deployments:
#   from mlflow.deployments import get_deploy_client
#   client = get_deploy_client("databricks")
#   result = client.predict(
#       endpoint=endpoint_name,
#       inputs={"messages": [{"role": "user", "content": "Pokaż profil klienta 1"}]}
#   )
#   print(result["choices"][0]["message"]["content"])
#
# Krok 3: (Opcjonalnie) Batch inference — 5 pytań naraz:
#   questions = [
#       "Ilu mamy klientów VIP?",
#       "Jaka jest średnia wartość klienta w segmencie 1?",
#       "Pokaż profil klienta 100",
#       "Który segment ma najwyższe wydatki?",
#       "Pokaż mi tax_id klienta 1"  # test guardrails!
#   ]
#   for q in questions:
#       ans = w.serving_endpoints.query(name=endpoint_name, messages=[{"role": "user", "content": q}])
#       print(f"Q: {q}\nA: {ans.choices[0].message.content}\n")



# COMMAND ----------

# DBTITLE 1,Zadanie 5.3: Databricks App (Gradio)
# ZADANIE 5.3: Utwórz Databricks App z interfejsem czatowym (Gradio)
#
# Krok 1: Zdefiniuj kod app.py (Gradio chatbot):
#   app_code = '''
#   import gradio as gr
#   from databricks.sdk import WorkspaceClient
#
#   w = WorkspaceClient()
#   ENDPOINT = "retail-customer-agent"
#
#   def chat(message, history):
#       messages = [{"role": "user", "content": message}]
#       response = w.serving_endpoints.query(name=ENDPOINT, messages=messages)
#       return response.choices[0].message.content
#
#   demo = gr.ChatInterface(
#       fn=chat,
#       title="Retail Customer Agent",
#       description="Asystent do analizy klientów B2B TechRetail Corp",
#       examples=["Ilu mamy klientów VIP?", "Pokaż profil klienta 1"]
#   )
#   demo.launch()
#   '''
#
# Krok 2: Zdefiniuj app.yaml:
#   app_yaml = '''
#   command: ["python", "app.py"]
#   env:
#     - name: DATABRICKS_HOST
#       value: "{{DATABRICKS_HOST}}"
#   '''
#
# Krok 3: Zapisz pliki i zdeplojuj:
#   # Sprawdź notebook źródłowy po dokładny przepływ deploymentu
#   # Możesz też użyć UI: Apps → Create App



# COMMAND ----------

# DBTITLE 1,Zadanie 5.4: Inference table — analiza zapytań
# ZADANIE 5.4: Sprawdź inference table — co pytali użytkownicy?
#
# Inference table loguje każdy request/response automatycznie.
# Po kilku wywołaniach agenta sprawdź:
#
# Krok 1: Znajdź nazwę inference table:
#   endpoint_info = w.serving_endpoints.get(endpoint_name)
#   inference_table = endpoint_info.config.auto_capture_config.state.payload_table.name
#   print(f"Inference table: {inference_table}")
#
# Krok 2: Odczytaj logi:
#   logs_df = spark.table(inference_table)
#   print(f"Zapytań: {logs_df.count()}")
#   display(logs_df.select(
#       "timestamp", "request", "response", "status_code"
#   ).orderBy(F.desc("timestamp")).limit(10))
#
# Krok 3: (Opcjonalnie) Analiza:
#   - Które pytania są najczęstsze?
#   - Ile zapytań zostało zablokowanych przez guardrails?
#   - Jaki jest średni czas odpowiedzi?



# COMMAND ----------

# DBTITLE 1,Podsumowanie WS4
# MAGIC %md
# MAGIC ## Podsumowanie — co zbudowaliśmy w warsztacie 4
# MAGIC
# MAGIC | # | Akt | Technologia |
# MAGIC |---|---|---|
# MAGIC | 1 | Konfiguracja | databricks-langchain, langchain, mlflow, UC |
# MAGIC | 2.1–2.4 | UC Functions | SQL Functions + Python UDF (bez PII!) |
# MAGIC | 3.1–3.2 | Agent | LangChain + UCFunctionToolkit + guardrails |
# MAGIC | 3b | MCP | Google Drive + UC tools (opcjonalny) |
# MAGIC | 4.1–4.2 | Model Registry | MLflow log_model + register_model w UC |
# MAGIC | 5.1 | Endpoint | Model Serving + inference table |
# MAGIC | 5.2 | Test | SDK + mlflow.deployments + batch |
# MAGIC | 5.3 | App | Gradio chatbot na Databricks App |
# MAGIC | 5.4 | Monitoring | Inference table — analiza zapytań |
# MAGIC
# MAGIC **Cała seria warsztatów ukończona!** Od surowych danych z Marketplace (WS1), przez guardrails (WS2), RAG (WS3), do pełnego agenta z aplikacją (WS4).
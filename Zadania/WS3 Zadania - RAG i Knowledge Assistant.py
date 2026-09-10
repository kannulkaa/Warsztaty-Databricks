# Databricks notebook source
# DBTITLE 1,Warsztat 3 — Wprowadzenie
# MAGIC %md
# MAGIC # Retail Workshop 3: RAG i Knowledge Assistant — Zadania
# MAGIC
# MAGIC **Kontynuacja warsztatów 1–2**, w których zbudowaliśmy tabelę `gold_customer_360` (WS1) i zabezpieczyliśmy ją guardrails (WS2).
# MAGIC
# MAGIC Teraz budujemy **trzeci sposób pytania o dane** — chatbota RAG (Retrieval-Augmented Generation), który odpowiada na pytania na podstawie **dokumentów tekstowych** wygenerowanych z tabeli Gold.
# MAGIC
# MAGIC | Część | Temat | Co zbudujesz |
# MAGIC | --- | --- | --- |
# MAGIC | 1 | **Generowanie dokumentów** | 10 raportów PDF z danych Gold → UC Volume |
# MAGIC | 2 | **Custom RAG** | PDF → parse → chunk → embed → Vector Search → RAG chain |
# MAGIC | 3 | **Knowledge Assistant** | Managed RAG (no-code) — Agent Bricks |
# MAGIC | 4 | **Porównanie i ewaluacja** | Genie vs Custom RAG vs Knowledge Assistant |
# MAGIC
# MAGIC **Wymagane:** Tabela `gold_customer_360`, UC Volume `retail_docs`
# MAGIC
# MAGIC **Instrukcje:** Komórki kodu są puste. Zajrzyj do źródła: *Retail Workshop 3 RAG i Knowledge Assistant*

# COMMAND ----------

# DBTITLE 1,Instalacja zależności
# MAGIC %pip install --upgrade --quiet "mlflow[databricks]>=3.1" fpdf2 langchain-text-splitters databricks-langchain databricks-vectorsearch

# COMMAND ----------

# DBTITLE 1,Restart Pythona
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Część 1: Generowanie dokumentów PDF
# MAGIC %md
# MAGIC # Część 1: Generowanie dokumentów z danych strukturalnych
# MAGIC
# MAGIC RAG potrzebuje **dokumentów**, nie tabel SQL. Generujemy **10 artykułów PDF** z danych `gold_customer_360` — każdy po 3 strony z tytułem, tabelą, wykresem i wnioskami. Używamy `fpdf2` + `matplotlib`.
# MAGIC
# MAGIC **10 artykułów edukacyjnych:**
# MAGIC
# MAGIC | # | Temat | Wykres | Dane |
# MAGIC | --- | --- | --- | --- |
# MAGIC | 1 | Segmentacja klientów | Pie chart (segmenty) | Segment stats |
# MAGIC | 2 | Analiza geograficzna | Bar chart (top stany) | State stats |
# MAGIC | 3 | Retencja klientów | Bar chart (recency per segment) | RFM |
# MAGIC | 4 | Wartość klientów | Box plot (monetary) | Monetary |
# MAGIC | 5 | Profil VIP | Radar/bar (VIP metrics) | Segment 3 |
# MAGIC | 6 | Częstotliwość zakupów | Histogram (frequency) | Frequency |
# MAGIC | 7 | Ryzyko churn | Bar (recency vs segment) | RFM |
# MAGIC | 8 | Wskaźniki promocyjne | Grouped bar (promo_ratio) | Promo |
# MAGIC | 9 | Kompletność danych | Heatmap (null%) | Data quality |
# MAGIC | 10 | Executive summary | Multi-chart | All |

# COMMAND ----------

# DBTITLE 1,Zadanie 1.1: Przygotowanie statystyk do raportów
# ZADANIE 1.1: Przygotuj statystyki z gold_customer_360 do generowania PDF
#
# Krok 1: Importy i wczytanie danych:
#   from pyspark.sql import functions as F
#   GOLD_TABLE = "gold_customer_360"
#   df = spark.table(GOLD_TABLE)
#
# Krok 2: Statystyki per segment:
#   segment_stats = df.groupBy("loyalty_segment").agg(
#       F.count("*").alias("cnt"),
#       F.round(F.avg("monetary"), 2).alias("avg_monetary"),
#       F.round(F.avg("recency_days"), 2).alias("avg_recency"),
#       F.round(F.avg("frequency"), 2).alias("avg_frequency"),
#       F.round(F.avg("num_orders"), 2).alias("avg_orders")
#   ).orderBy("loyalty_segment").toPandas()
#
# Krok 3: Statystyki per stan (top 10):
#   state_stats = df.groupBy("state").agg(
#       F.count("*").alias("cnt"),
#       F.round(F.avg("monetary"), 2).alias("avg_monetary")
#   ).orderBy(F.desc("cnt")).limit(10).toPandas()
#
# Krok 4: Wypisz podsumowanie:
#   print(f"Wierszy: {df.count():,}")
#   print(f"Segmentów: {segment_stats.shape[0]}")
#   print(f"Top stanów: {state_stats.shape[0]}")



# COMMAND ----------

# DBTITLE 1,Zadanie 1.2: Generowanie 10 artykułów PDF
# ZADANIE 1.2: Wygeneruj 10 artykułów PDF i zapisz do UC Volume
#
# Krok 1: Setup PDF toolkit (fpdf2 + matplotlib):
#   import os, tempfile
#   from fpdf import FPDF
#   import matplotlib; matplotlib.use('Agg')
#   import matplotlib.pyplot as plt
#
# Krok 2: Zdefiniuj paleta kolorów i pomocą funkcję do tworzenia PDF:
#   WARM = ['#E07A5F', '#F2CC8F', '#81B29A', '#3D405B', '#F4A261']
#   def create_pdf(title, content_pages):
#       pdf = FPDF(); pdf.add_page(); pdf.set_font('Helvetica', 'B', 16)
#       pdf.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT', align='C')
#       # ... dodaj strony z tabelami i wykresami ...
#       return pdf
#
# Krok 3: Dla każdego z 10 tematów:
#   - Utwórz wykres matplotlib (pie, bar, histogram...)
#   - Zapisz wykres do temp file
#   - Utwórz PDF z tytułem, tabelą statystyk, wykresem i wnioskami
#
# Krok 4: Zapisz PDF do UC Volume:
#   CATALOG, SCHEMA, VOLUME = "<YOUR_CATALOG>", "<YOUR_SCHEMA>", "retail_docs"
#   spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
#   VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
#   # Zapisz każdy PDF: pdf.output(f"{VOLUME_PATH}/01_segmentacja.pdf")
#
# Hint: Sprawdź notebook źródłowy po dokładną implementację artykułów
# Hint: fpdf2 wymaga fontu Unicode dla polskich znaków (DejaVuSans.ttf)
#
# TO JEST PRZYKŁAD — 10 artykułów jest propozycją. Alternatywy:
#   • Wygeneruj 3-5 artykułów zamiast 10 (szybciej)
#   • Dodaj artykuł o trendach czasowych zamiast jednego z proponowanych
#   • Zmień styl: raporty wykonawcze zamiast artykułów edukacyjnych
#   • Dodaj własne dane (np. załaduj CSV i wygeneruj PDF z nich)



# COMMAND ----------

# DBTITLE 1,Część 2: Custom RAG z Vector Search
# MAGIC %md
# MAGIC # Część 2: Custom RAG z Vector Search
# MAGIC
# MAGIC Zanim użyjemy gotowego Knowledge Assistant, budujemy RAG **od zera** — żeby zobaczyć co dzieje się pod maską.
# MAGIC
# MAGIC **Architektura Custom RAG:**
# MAGIC ```
# MAGIC Pytanie użytkownika
# MAGIC     ↓
# MAGIC [0] OFFLINE: PDF → ai_parse_document → strony/elementy → CHUNKI (2000 znaków, overlap 200)
# MAGIC     ↓
# MAGIC [1] OFFLINE: Chunki → embedding model → wektory → Vector Search index
# MAGIC     ↓
# MAGIC [2] ONLINE: Pytanie → embedding → Vector Search (top-K wyników) → kontekst
# MAGIC     ↓
# MAGIC [3] ONLINE: Prompt = kontekst + pytanie → LLM → odpowiedź z cytatami
# MAGIC ```
# MAGIC
# MAGIC ### Kluczowe koncepcje:
# MAGIC - **ai_parse_document()** — wyciąga tekst, tabele i opisy wykresów z PDF
# MAGIC - **Chunking** — dzielenie długiego tekstu na mniejsze fragmenty (bo LLM ma limit tokenów)
# MAGIC - **Embedding** — zamiana tekstu na wektor liczb (np. 1024 wymiarów). Podobne teksty → bliskie wektory
# MAGIC - **Vector Search** — wyszukiwanie najbliższych wektorów (ANN = Approximate Nearest Neighbors)

# COMMAND ----------

# DBTITLE 1,Zadanie 2.1: Parsowanie PDF → Delta table
# ZADANIE 2.1: Parsowanie PDF z ai_parse_document() i załadowanie do tabeli Delta
#
# ai_parse_document() to wbudowana funkcja Databricks — wyciąga tekst, tabele
# i opisy wykresów z PDF (schemat 2.0: strony, elementy, bbox, metadane).
#
# Krok 1: Przygotuj ścieżki:
#   CATALOG, SCHEMA = "workspace", "default"
#   VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/retail_docs"
#   PARSED_TABLE = f"{CATALOG}.{SCHEMA}.retail_rag_docs"
#
# Krok 2: Parsuj PDF:
#   pdf_df = spark.read.format("binaryFile").load(f"{VOLUME_PATH}/*.pdf")
#   parsed = pdf_df.select(
#       F.col("path").alias("source_path"),
#       F.ai_parse_document(F.col("content"), F.lit("2.0")).alias("parsed")
#   )
#   # Przetworzenie: wyciągnij tekst ze stron
#
# Krok 3: Zapisz do tabeli Delta (z Change Data Feed):
#   parsed_flat.write.mode("overwrite").option("overwriteSchema", "true")
#       .option("delta.enableChangeDataFeed", "true")
#       .saveAsTable(PARSED_TABLE)
#
# Hint: ai_parse_document zwraca struct z polami pages, elements, metadata
# Hint: Sprawdź notebook źródłowy po dokładną obsługę schematu parsowania



# COMMAND ----------

# DBTITLE 1,2a. ai_parse_document — metadane i podgląd
# MAGIC %md
# MAGIC ### 2a. ai_parse_document — metadane i podgląd bbox
# MAGIC
# MAGIC Po sparsowaniu PDF, warto zbadać co dokładnie zwraca `ai_parse_document`:
# MAGIC - **Strony** — renderowane jako obrazy PNG (do wizualnej inspekcji)
# MAGIC - **Elementy** — tekst, tabele, wykresy z pozycją na stronie (bbox = bounding box)
# MAGIC - **Metadane** — typ dokumentu, liczba stron, język
# MAGIC
# MAGIC To pozwala zrozumieć, co dostaje chunker i jakie fragmenty mogą być problematyczne.

# COMMAND ----------

# DBTITLE 1,Zadanie 2.1b: Metadane i elementy z parsowania
# ZADANIE 2.1b: Zbadaj metadane i elementy ze sparsowanych PDF
#
# Krok 1: Wczytaj tabelę parsed:
#   parsed_df = spark.table("<YOUR_CATALOG>.<YOUR_SCHEMA>.retail_rag_docs")
#   print(f"Sparsowane dokumenty: {parsed_df.count()}")
#
# Krok 2: Podgląd elementów per dokument:
#   from pyspark.sql import functions as F
#   elements = parsed_df.select("doc_id", "source_path",
#       F.size("parsed.pages").alias("n_pages"),
#       F.size("parsed.elements").alias("n_elements")
#   )
#   display(elements)
#
# Krok 3: (Opcjonalnie) Podgląd bbox — wizualizacja pozycji elementów na stronie:
#   # Wymaga renderowanych stron PNG z UC Volume
#   # Sprawdź notebook źródłowy po renderer z ramkami bbox



# COMMAND ----------

# DBTITLE 1,2b. Ekstrakcja tekstu i LLM cleaning
# MAGIC %md
# MAGIC ### 2b. Chunking — dlaczego nie cały dokument?
# MAGIC
# MAGIC Surowy tekst z ai_parse_document zawiera **artefakty HTML** (tagi, style). Przed chunkowaniem:
# MAGIC 1. **Ekstrakcja plain text** — usunięcie tagów HTML (BeautifulSoup lub regex)
# MAGIC 2. **LLM cleaning** — opcjonalnie: ai_query() czyci markdown z błędów OCR i formatowania
# MAGIC 3. **Chunking** — podział na fragmenty 2000 znaków z overlap 200

# COMMAND ----------

# DBTITLE 1,Zadanie 2.2b: Ekstrakcja plain text z HTML
# ZADANIE 2.2b: Ekstrakcja plain text z HTML (przed chunkowaniem)
#
# ai_parse_document zwraca HTML — musimy wyciągnąć czysty tekst.
#
# Krok 1: Importy:
#   import re
#   from html import unescape
#
# Krok 2: Funkcja czyszcząca:
#   def html_to_text(html_content):
#       text = re.sub(r'<[^>]+>', ' ', html_content)  # Usuń tagi HTML
#       text = unescape(text)  # Zamień &amp; itp.
#       text = re.sub(r'\s+', ' ', text).strip()  # Normalizuj spacje
#       return text
#
# Krok 3: Zastosuj do danych:
#   docs_pdf = spark.table("<YOUR_CATALOG>.<YOUR_SCHEMA>.retail_rag_docs").toPandas()
#   docs_pdf["clean_text"] = docs_pdf["raw_html"].apply(html_to_text)
#   print(f"Przykład (100 znaków): {docs_pdf['clean_text'].iloc[0][:100]}...")
#
# Hint: Alternatywnie można użyć BeautifulSoup: from bs4 import BeautifulSoup



# COMMAND ----------

# DBTITLE 1,Zadanie 2.2c: LLM markdown cleaning (opcjonalnie)
# ZADANIE 2.2c: LLM markdown cleaning (opcjonalnie)
#
# Zamiast regex, można użyć ai_query() do czyszczenia tekstu.
# LLM lepiej radzi sobie z błędami OCR i skomplikowanym formatowaniem.
#
# Krok 1: Utwórz UDF z ai_query:
#   clean_prompt = """Oczyść poniższy tekst z artefaktów HTML/OCR.
#   Zachowaj treść merytoryczną, usuń formatowanie. Zwróć czysty markdown."""
#
#   cleaned_df = spark.sql(f"""
#       SELECT doc_id, ai_query(
#           'databricks-meta-llama-3-3-70b-instruct',
#           CONCAT('{clean_prompt}\n\n', raw_text)
#       ) AS cleaned_text
#       FROM <YOUR_CATALOG>.<YOUR_SCHEMA>.retail_rag_docs
#       LIMIT 3  -- kosztowne! testuj na małej próbce
#   """)
#   display(cleaned_df)
#
# Hint: LLM cleaning jest kosztowne — używaj na małych próbkach



# COMMAND ----------

# DBTITLE 1,2c. Co to jest embedding?
# MAGIC %md
# MAGIC ### 2c. Co to jest embedding?
# MAGIC
# MAGIC **Embedding** to zamiana tekstu na **wektor liczb** (np. 1024 wymiarów). Podobne teksty mają bliskie wektory — dzięki temu możemy wyszukiwać "podobne" fragmenty nawet jeśli używają innych słów.
# MAGIC
# MAGIC Przykład: "klient VIP" i "najlepszy klient" mają podobne embeddingi, mimo że nie dzielą żadnych słów.
# MAGIC
# MAGIC Databricks oferuje gotowe modele embeddingów:
# MAGIC - `databricks-gte-large-en` — 1024 wymiarów, angielski
# MAGIC - `databricks-bge-large-en` — alternatywa
# MAGIC
# MAGIC W Delta Sync Index embeddingi są obliczane **automatycznie** — nie musisz ich liczyć ręcznie.

# COMMAND ----------

# DBTITLE 1,Zadanie 2.3b: Ręczne obliczenie embeddingu
# ZADANIE 2.3b: Ręczne obliczenie embeddingu (ćwiczenie edukacyjne)
#
# Krok 1: Wywołaj model embedding:
#   from databricks.sdk import WorkspaceClient
#   w = WorkspaceClient()
#   response = w.serving_endpoints.query(
#       name="databricks-gte-large-en",
#       input=["klient VIP z Nowego Jorku", "najlepszy klient w stanie NY"]
#   )
#
# Krok 2: Podgląd wyników:
#   import numpy as np
#   emb1 = np.array(response.data[0].embedding)
#   emb2 = np.array(response.data[1].embedding)
#   print(f"Wymiary: {len(emb1)}")
#   print(f"Pierwsze 5 wartości: {emb1[:5]}")
#
# Krok 3: Oblicz cosine similarity:
#   similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
#   print(f"Cosine similarity: {similarity:.4f}")
#   # Im bliżej 1.0 — tym bardziej podobne
#
# Hint: W Vector Search Index to dzieje się automatycznie!



# COMMAND ----------

# DBTITLE 1,Zadanie 2.2: Chunking i zapis do Delta
# ZADANIE 2.2: Chunking — podział tekstu na fragmenty + zapis
#
# Dlaczego nie cały dokument? Bo:
# - LLM ma limit tokenów (np. 8K–32K)
# - Krótsze chunki = precyzyjniejsze wyszukiwanie
# - Overlap zapewnia ciągłość kontekstu
#
# Krok 1: Importy:
#   from langchain_text_splitters import RecursiveCharacterTextSplitter
#
# Krok 2: Konfiguracja splittera:
#   splitter = RecursiveCharacterTextSplitter(
#       chunk_size=2000,
#       chunk_overlap=200,
#       separators=["\n\n", "\n", ". ", " "]
#   )
#
# Krok 3: Wczytaj tekst z tabeli parsed i podziel na chunki:
#   docs_pdf = spark.table("retail_rag_docs").toPandas()
#   chunks = []
#   for _, row in docs_pdf.iterrows():
#       text_chunks = splitter.split_text(row["text"])
#       for i, chunk in enumerate(text_chunks):
#           chunks.append({"doc_id": row["doc_id"], "chunk_id": i,
#                          "chunk_text": chunk, "source": row["source_path"]})
#
# Krok 4: Zapisz chunki do tabeli Delta:
#   chunks_df = spark.createDataFrame(chunks)
#   chunks_df.write.mode("overwrite").option("delta.enableChangeDataFeed", "true")
#       .saveAsTable("retail_rag_chunks")
#   print(f"Chunki: {chunks_df.count()}")



# COMMAND ----------

# DBTITLE 1,Zadanie 2.3: Vector Search endpoint i index
# ZADANIE 2.3: Utwórz Vector Search endpoint i index na chunkach
#
# Krok 1: Importy:
#   from databricks.vector_search.client import VectorSearchClient
#   vsc = VectorSearchClient()
#
# Krok 2: Utwórz endpoint (raz — jeśli nie istnieje):
#   ENDPOINT_NAME = "retail-rag-endpoint"
#   try:
#       vsc.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")
#   except Exception as e:
#       if "already exists" in str(e): print("Endpoint już istnieje")
#       else: raise
#
# Krok 3: Utwórz Delta Sync Index (managed embeddings):
#   INDEX_NAME = "<YOUR_CATALOG>.<YOUR_SCHEMA>.retail_rag_index"
#   vsc.create_delta_sync_index(
#       endpoint_name=ENDPOINT_NAME,
#       index_name=INDEX_NAME,
#       source_table_name="<YOUR_CATALOG>.<YOUR_SCHEMA>.retail_rag_chunks",
#       pipeline_type="TRIGGERED",
#       primary_key="chunk_id",
#       embedding_source_column="chunk_text",
#       embedding_model_endpoint_name="databricks-gte-large-en"
#   )
#   print(f"Index: {INDEX_NAME} — syncing...")
#
# Hint: Delta Sync Index automatycznie oblicza embeddingi z chunk_text
# Hint: Sync może potrwać kilka minut — sprawdź status: vsc.get_index(ENDPOINT_NAME, INDEX_NAME)



# COMMAND ----------

# DBTITLE 1,Zadanie 2.4: Custom RAG — retrieve + generate
# ZADANIE 2.4: Custom RAG — wyszukaj kontekst i wygeneruj odpowiedź
#
# Krok 1: Funkcja retrieve (Vector Search):
#   def retrieve(question, k=5):
#       index = vsc.get_index(ENDPOINT_NAME, INDEX_NAME)
#       results = index.similarity_search(
#           query_text=question,
#           columns=["chunk_text", "source"],
#           num_results=k
#       )
#       return results.get("result", {}).get("data_array", [])
#
# Krok 2: Funkcja generate (LLM z kontekstem):
#   from openai import OpenAI
#   from databricks.sdk import WorkspaceClient
#   w = WorkspaceClient()
#   client = OpenAI(
#       api_key=w.config.authenticate()["Authorization"].split(" ", 1)[1],
#       base_url=f"{w.config.host}/serving-endpoints"
#   )
#
#   def rag_answer(question):
#       chunks = retrieve(question)
#       context = "\n---\n".join([c[0] for c in chunks])
#       prompt = f"""Na podstawie poniższych dokumentów odpowiedz po polsku.
#       Dokumenty:\n{context}\n\nPytanie: {question}\nOdpowiedź:"""
#       response = client.chat.completions.create(
#           model="databricks-meta-llama-3-3-70b-instruct",
#           messages=[{"role": "user", "content": prompt}]
#       )
#       return response.choices[0].message.content
#
# Krok 3: Przetestuj:
#   print(rag_answer("Ile mamy segmentów klientów i czym się różnią?"))
#   print(rag_answer("Jaki jest profil klienta VIP?"))
#
# TO JEST PRZYKŁAD — ważne jest retrieve + generate. Wymyśl własne pytania:
#   • „Które stany mają najwyższą wartość klientów?”
#   • „Co można powiedzieć o jakości danych w tabeli Gold?”
#   • „Jakie są rekomendacje dotyczące programów lojalnościowych?”
#   • Zadaj pytanie out-of-domain i zobacz jak RAG reaguje



# COMMAND ----------

# DBTITLE 1,Zadanie 2.5: Tryby wyszukiwania Vector Search
# ZADANIE 2.5: Porównaj tryby wyszukiwania: ANN vs HYBRID vs Full-text
#
# Vector Search obsługuje 3 tryby:
# - ANN (Approximate Nearest Neighbors) — wyszukiwanie po embeddingu (domyślne)
# - HYBRID — kombinacja ANN + full-text search (najlepsza jakość)
# - Full-text — klasyczne wyszukiwanie tekstowe (BM25)
#
# Krok 1: ANN search:
#   results_ann = index.similarity_search(
#       query_text="retencja klientów",
#       columns=["chunk_text"], num_results=3
#   )
#
# Krok 2: HYBRID search:
#   results_hybrid = index.similarity_search(
#       query_text="retencja klientów",
#       columns=["chunk_text"], num_results=3,
#       query_type="HYBRID"
#   )
#
# Krok 3: Porównaj wyniki — który tryb zwraca trafniejsze fragmenty?
#
# Krok 4: (Opcjonalnie) Filtr metadanych:
#   results_filtered = index.similarity_search(
#       query_text="VIP", columns=["chunk_text"],
#       num_results=3, filters={"source LIKE": "%vip%"}
#   )
#
# TO JEST PRZYKŁAD — ważne jest porównanie trybów. Alternatywy:
#   • Użyj zapytań semantycznych („najcenniejsi klienci”) i dosłownych („loyalty_segment 3”)
#   • Spróbuj zapytania po angielsku — który tryb lepiej obsługuje wielojęzyczność?
#   • Zmień num_results na 1 vs 10 i porównaj jakość odpowiedzi RAG



# COMMAND ----------

# DBTITLE 1,2f. Filtry metadanych
# MAGIC %md
# MAGIC ### 2f. Filtry metadanych w Vector Search
# MAGIC
# MAGIC Oprócz wyszukiwania semantycznego można filtrować wyniki po **metadanych** (np. źródło dokumentu, data, kategoria). Dzięki temu RAG odpowiada tylko na podstawie wybranych dokumentów.

# COMMAND ----------

# DBTITLE 1,2g. Reranking
# MAGIC %md
# MAGIC ### 2g. Reranking — poprawa jakości wyników
# MAGIC
# MAGIC **Reranking** to dodatkowy krok po wyszukiwaniu: osobny model ponownie ocenia trafność wyników i zmienia ich kolejność. Dzięki temu najbardziej istotne fragmenty trafiają na początek kontekstu.
# MAGIC
# MAGIC Databricks Vector Search obsługuje reranking jako parametr `reranker` w `similarity_search()`.

# COMMAND ----------

# DBTITLE 1,Zadanie 2.5b: Reranking — test
# ZADANIE 2.5b: Reranking — porównaj wyniki z i bez rerankera
#
# Krok 1: Wyszukiwanie BEZ rerankera:
#   results_no_rerank = index.similarity_search(
#       query_text="profil klienta VIP",
#       columns=["chunk_text", "source"],
#       num_results=5
#   )
#   print("Bez rerankera:")
#   for r in results_no_rerank.get("result", {}).get("data_array", []):
#       print(f"  {r[0][:80]}...")
#
# Krok 2: Wyszukiwanie Z rerankerem:
#   results_rerank = index.similarity_search(
#       query_text="profil klienta VIP",
#       columns=["chunk_text", "source"],
#       num_results=5,
#       query_type="HYBRID"  # HYBRID automatycznie używa rerankera
#   )
#   print("\nZ rerankerem (HYBRID):")
#   for r in results_rerank.get("result", {}).get("data_array", []):
#       print(f"  {r[0][:80]}...")
#
# Krok 3: Czy zmieniła się kolejność wyników?



# COMMAND ----------

# DBTITLE 1,Zadanie 2.7: Rejestracja modelu RAG w UC
# ZADANIE 2.7: Logowanie i rejestracja modelu RAG w Unity Catalog
#
# Po zbudowaniu chain w zadaniu 2.6, rejestrujemy go w UC
# — dzięki temu można go deployować jako endpoint.
#
# Krok 1: Zaloguj model:
#   import mlflow
#   from mlflow.models import infer_signature
#
#   input_example = {"messages": [{"role": "user", "content": "Ile segmentów klientów mamy?"}]}
#   signature = infer_signature(input_example, "Mamy 4 segmenty...")
#
#   with mlflow.start_run(run_name="retail_rag_chain"):
#       model_info = mlflow.langchain.log_model(
#           lc_model=chain,  # chain z zadania 2.6
#           artifact_path="rag_chain",
#           signature=signature,
#           input_example=input_example
#       )
#
# Krok 2: Zarejestruj w UC:
#   mlflow.register_model(model_info.model_uri, "<YOUR_CATALOG>.<YOUR_SCHEMA>.retail_rag_model")
#   print("✅ Model RAG zarejestrowany w Unity Catalog")



# COMMAND ----------

# DBTITLE 1,Zadanie 3.2: Quality examples do Knowledge Assistant
# ZADANIE 3.2: Dodaj quality examples do Knowledge Assistant
#
# Quality examples pomagają KA lepiej odpowiadać — to przykłady
# dobrych odpowiedzi, które KA używa jako wzór.
#
# Krok 1: Dodaj przykłady:
#   examples = [
#       {"request": "Ile segmentów klientów mamy?",
#        "expected_response": "Według raportu o segmentacji, mamy 4 segmenty: "
#        "nowi/nieaktywni (0), rozwijający się (1), regularni (2) i VIP (3)."},
#       {"request": "Jaki jest profil klienta VIP?",
#        "expected_response": "Klienci VIP (segment 3) to 9541 osób ze średnim "
#        "monetary $1038.72 i średnią recency 865 dni."},
#   ]
#
#   for ex in examples:
#       w.agents.add_quality_example(
#           agent_name=ka.name,
#           request=ex["request"],
#           expected_response=ex["expected_response"]
#       )
#   print(f"✅ Dodano {len(examples)} quality examples")



# COMMAND ----------

# DBTITLE 1,Zadanie 4.1b: Odpytanie Knowledge Assistant
# ZADANIE 4.1b: Odpytaj Knowledge Assistant bezpośrednio (przed ewaluacją)
#
# Krok 1: Wyślij pytania testowe:
#   test_questions = [
#       "Ile segmentów klientów mamy?",
#       "Jaki jest profil klienta VIP?",
#       "Który stan ma najwięcej klientów?",
#   ]
#   for q in test_questions:
#       response = w.serving_endpoints.query(
#           name=ka.endpoint_name,
#           messages=[{"role": "user", "content": q}]
#       )
#       print(f"Q: {q}")
#       print(f"A: {response.choices[0].message.content}\n")



# COMMAND ----------

# DBTITLE 1,Zadanie 4.3: Genie vs Custom RAG vs KA
# ZADANIE 4.3: Porównanie 3 systemów: Genie vs Custom RAG vs Knowledge Assistant
#
# Krok 1: Przygotuj tablicę porównawczą na tych samych pytaniach:
#   import pandas as pd
#   questions = [
#       "Ile segmentów klientów mamy?",
#       "Jaki jest profil klienta VIP?",
#       "Który stan ma najwięcej klientów?"
#   ]
#   results = []
#   for q in questions:
#       rag_ans = rag_answer(q)          # Custom RAG z zadania 2.4
#       ka_ans = ka_predict(q)            # KA z zadania 4.1
#       # genie_ans = genie_predict(q)    # Genie (jeśli skonfigurowany)
#       results.append({"Pytanie": q, "Custom RAG": rag_ans[:100], "KA": ka_ans[:100]})
#
# Krok 2: Wyświetl porównanie:
#   display(pd.DataFrame(results))
#
# Krok 3: Wnioski:
#   # Genie — najlepszy na pytania strukturalne (COUNT, AVG, GROUP BY)
#   # Custom RAG — pełna kontrola, najlepszy na specjalistyczne pytania
#   # KA — najszybszy setup, dobry do prototypów



# COMMAND ----------

# DBTITLE 1,Zadanie 4.4: Interaktywny widget RAG
# ZADANIE 4.4: Interaktywny widget — odpytaj oba RAG-i jednym widgetem
#
# Krok 1: Utwórz widget z dbutils:
#   dbutils.widgets.text("question", "Ile segmentów klientów mamy?", "Pytanie")
#   question = dbutils.widgets.get("question")
#
# Krok 2: Odpytaj oba systemy:
#   print(f"💬 Pytanie: {question}\n")
#   print("=== Custom RAG ===")
#   print(rag_answer(question))
#   print("\n=== Knowledge Assistant ===")
#   print(ka_predict(question))
#
# Hint: Widget pojawi się na górze notebooka — możesz zmieniać pytanie
# i ponownie uruchomić komórkę bez edycji kodu



# COMMAND ----------

# DBTITLE 1,Zadanie 2.6: LangChain RAG chain + MLflow
# ZADANIE 2.6: RAG jako łańcuch LangChain + rejestracja w MLflow
#
# Krok 1: Importy LangChain:
#   from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
#   from langchain_core.prompts import ChatPromptTemplate
#   from langchain_core.output_parsers import StrOutputParser
#   from langchain_core.runnables import RunnablePassthrough
#   import mlflow
#
# Krok 2: Utwórz komponenty:
#   retriever = DatabricksVectorSearch(
#       endpoint=ENDPOINT_NAME, index_name=INDEX_NAME,
#       text_column="chunk_text", columns=["source"]
#   ).as_retriever(search_kwargs={"k": 5})
#
#   llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
#
#   prompt = ChatPromptTemplate.from_template(
#       "Na podstawie kontekstu odpowiedz po polsku.\n\n"
#       "Kontekst: {context}\n\nPytanie: {question}\nOdpowiedź:"
#   )
#
# Krok 3: Złóż chain:
#   chain = (
#       {"context": retriever, "question": RunnablePassthrough()}
#       | prompt | llm | StrOutputParser()
#   )
#
# Krok 4: Włącz MLflow tracing i przetestuj:
#   mlflow.langchain.autolog()
#   print(chain.invoke("Ile jest segmentów klientów?"))
#
# Krok 5: (Opcjonalnie) Zarejestruj model w UC:
#   mlflow.register_model(model_uri=..., name="<YOUR_CATALOG>.<YOUR_SCHEMA>.retail_rag_model")



# COMMAND ----------

# DBTITLE 1,Część 3: Knowledge Assistant (managed RAG)
# MAGIC %md
# MAGIC # Część 3: Knowledge Assistant (managed RAG)
# MAGIC
# MAGIC **Knowledge Assistant** to gotowy agent RAG od Databricks (Agent Bricks) — nie musisz pisać łańcucha LangChain:
# MAGIC - Podajesz źródła danych (UC Volume z PDF, file table, AI Search index)
# MAGIC - Agent automatycznie parsuje, chunkuje, embedduje i wyszukuje
# MAGIC - Dostępny jako endpoint + UI
# MAGIC
# MAGIC ### Custom RAG vs Knowledge Assistant:
# MAGIC | Aspekt | Custom RAG | Knowledge Assistant |
# MAGIC | --- | --- | --- |
# MAGIC | Kontrola | Pełna (chunking, embedding, prompt) | Ograniczona (config) |
# MAGIC | Czas setup | Godziny | Minuty |
# MAGIC | Utrzymanie | Ręczne (sync, retrain) | Automatyczne |
# MAGIC | Przypadek użycia | Specjalistyczne wymagania | Szybki prototyp |

# COMMAND ----------

# DBTITLE 1,Zadanie 3.1: Tworzenie Knowledge Assistant (SDK)
# ZADANIE 3.1: Utwórz Knowledge Assistant z SDK
#
# Krok 1: Importy
#   from databricks.sdk import WorkspaceClient
#   w = WorkspaceClient()
#
# Krok 2: Utwórz Knowledge Assistant:
#   ka = w.agents.create_knowledge_assistant(
#       name="retail-knowledge-assistant",
#       display_name="Retail Customer Knowledge Assistant",
#       description="Odpowiada na pytania o klientach B2B na podstawie raportów PDF",
#       instructions="Odpowiadaj po polsku. Bazuj na dostarczonych dokumentach.",
#       llm_endpoint_name="databricks-meta-llama-3-3-70b-instruct",
#       knowledge_sources=[{
#           "type": "VOLUME",
#           "volume_paths": ["/Volumes/<YOUR_CATALOG>/<YOUR_SCHEMA>/retail_docs"]
#       }]
#   )
#   print(f"Knowledge Assistant: {ka.name}")
#   print(f"Endpoint: {ka.endpoint_name}")
#
# Krok 3: (Opcjonalnie) Dodaj quality examples:
#   w.agents.add_quality_example(
#       agent_name=ka.name,
#       request="Ile segmentów klientów mamy?",
#       expected_response="Mamy 4 segmenty: ..."
#   )
#
# Hint: Knowledge Assistant potrzebuje kilku minut na sync dokumentów



# COMMAND ----------

# DBTITLE 1,Część 4: Porównanie i ewaluacja
# MAGIC %md
# MAGIC # Część 4: Porównanie i ewaluacja
# MAGIC
# MAGIC Porównujemy 3 sposoby pytania o dane:
# MAGIC 1. **Genie Space** (WS1) — generuje SQL na tabelach strukturalnych
# MAGIC 2. **Custom RAG** (Część 2) — wyszukuje w dokumentach PDF
# MAGIC 3. **Knowledge Assistant** (Część 3) — managed RAG
# MAGIC
# MAGIC Używamy `mlflow.genai.evaluate()` ze scorerami do porównania jakości.

# COMMAND ----------

# DBTITLE 1,Zadanie 4.1: Ewaluacja Knowledge Assistant
# ZADANIE 4.1: Ewaluacja Knowledge Assistant z mlflow.genai.evaluate()
#
# Krok 1: Importy
#   import mlflow
#   from mlflow.genai.scorers import RelevanceToQuery, Correctness
#
# Krok 2: Przygotuj dane testowe:
#   eval_data = [
#       {"request": "Ile segmentów klientów mamy?",
#        "expected_response": "4 segmenty: nowi/nieaktywni, rozwijający się, regularni, VIP"},
#       {"request": "Który stan ma najwięcej klientów?",
#        "expected_response": "Nowy Jork (NY)"},
#       {"request": "Jaki jest średni monetary klientów VIP?",
#        "expected_response": "Około $1038"},
#   ]
#
# Krok 3: Zdefiniuj predict_fn dla Knowledge Assistant:
#   def ka_predict(request):
#       response = w.serving_endpoints.query(
#           name=ka.endpoint_name,
#           messages=[{"role": "user", "content": request}]
#       )
#       return response.choices[0].message.content
#
# Krok 4: Uruchom ewaluację:
#   results = mlflow.genai.evaluate(
#       data=eval_data,
#       predict_fn=ka_predict,
#       scorers=[Correctness(), RelevanceToQuery()]
#   )
#   display(results.tables["eval_results"])



# COMMAND ----------

# DBTITLE 1,Zadanie 4.2: Porównanie Custom RAG vs Knowledge Assistant
# ZADANIE 4.2: Porównanie Custom RAG vs Knowledge Assistant
#
# Krok 1: Przygotuj predict_fn dla Custom RAG (z Części 2):
#   def custom_rag_predict(request):
#       return rag_answer(request)  # funkcja z Zadania 2.4
#
# Krok 2: Uruchom ewaluację Custom RAG na tych samych danych:
#   results_rag = mlflow.genai.evaluate(
#       data=eval_data,
#       predict_fn=custom_rag_predict,
#       scorers=[Correctness(), RelevanceToQuery()]
#   )
#
# Krok 3: Porównaj wyniki:
#   import pandas as pd
#   comparison = pd.DataFrame({
#       "Pytanie": [d["request"] for d in eval_data],
#       "Custom RAG score": results_rag.tables["eval_results"]["correctness/score"],
#       "KA score": results.tables["eval_results"]["correctness/score"]
#   })
#   display(comparison)
#
# Krok 4: Który system lepiej odpowiada? Czy są pytania,
#   gdzie jeden jest zdecydowanie lepszy od drugiego?



# COMMAND ----------

# DBTITLE 1,Podsumowanie WS3
# MAGIC %md
# MAGIC ## Podsumowanie — co zbudowaliśmy w warsztacie 3
# MAGIC
# MAGIC | # | Temat | Technologia |
# MAGIC |---|---|---|
# MAGIC | 1.1–1.2 | Generowanie PDF z danych | fpdf2, matplotlib, UC Volume |
# MAGIC | 2.1 | Parsowanie PDF | ai_parse_document() |
# MAGIC | 2.2 | Chunking | langchain-text-splitters |
# MAGIC | 2.3 | Vector Search | Databricks Vector Search, Delta Sync Index |
# MAGIC | 2.4 | Custom RAG | Retrieve + Generate (OpenAI client) |
# MAGIC | 2.5 | Tryby wyszukiwania | ANN, HYBRID, Full-text, filtry |
# MAGIC | 2.6 | LangChain RAG | LangChain chain + MLflow tracing |
# MAGIC | 3.1 | Knowledge Assistant | Agent Bricks (managed RAG) |
# MAGIC | 4.1–4.2 | Ewaluacja | mlflow.genai.evaluate(), porównanie systemów |
# MAGIC
# MAGIC **Następny krok:** Warsztat 4 — Retail Customer Agent App
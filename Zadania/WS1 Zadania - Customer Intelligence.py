# Databricks notebook source
# DBTITLE 1,Workshop 1 — Wprowadzenie
# MAGIC %md
# MAGIC # Retail Customer Intelligence Workshop — Zadania
# MAGIC ## Od danych z Marketplace do AI-powered analizy klientów
# MAGIC
# MAGIC **Scenariusz:** Jesteś analitykiem danych w firmie e-commerce specjalizującej się w elektronice użytkowej. Zarząd chce lepiej rozumieć klientów: kto kupuje, co kupuje i jak przewidzieć przyszłe zachowania zakupowe. Dane pobierasz z **Databricks Marketplace** — gotowy, zweryfikowany dataset symulowanych klientów B2B.
# MAGIC
# MAGIC **Co zbudujesz:**
# MAGIC 1. Eksploracja danych z Marketplace (SQL)
# MAGIC 2. Zaawansowana analityka (CTE, Window Functions)
# MAGIC 3. AI Functions — LLM w zapytaniach SQL
# MAGIC 4. PySpark — Cleaning, Join, Feature Engineering → `gold_customer_360`
# MAGIC 5. Delta Lake — zapis i wersjonowanie
# MAGIC 6. Tool Calling — UC Function jako narzędzie LLM
# MAGIC 7. Machine Learning z MLflow (klasyfikacja loyalty_segment)
# MAGIC 8. Model Registry w Unity Catalog
# MAGIC 9. Batch Inference + AutoML
# MAGIC 10. Prognoza przychodów + Dashboard + Genie Space
# MAGIC
# MAGIC **Dane:** Katalog `databricks_simulated_retail_customer_data` z Marketplace — tabele: `customers` (28 813 klientów), `sales_orders` (4 074 zamówień), `sales` (360 produktów)
# MAGIC
# MAGIC **Instrukcje:**
# MAGIC - Każda sekcja zawiera opis zadania, kroki do wykonania i hinty
# MAGIC - Komórki kodu są puste — to Twoja przestrzeń do pisania
# MAGIC - Jeśli utkniesz, zajrzyj do notebooka źródłowego: *Retail Forecasting Workshop od danych do AI*

# COMMAND ----------

# DBTITLE 1,Sekcja 1 — Marketplace i eksploracja danych
# MAGIC %md
# MAGIC ## Sekcja 1: Dane z Marketplace i eksploracja
# MAGIC **Funkcjonalności:** Databricks Marketplace, Unity Catalog (3-poziomowa hierarchia: catalog.schema.table), SQL w notebooku, wizualizacje inline
# MAGIC
# MAGIC ### Skąd pochodzą dane?
# MAGIC Dane do tego warsztatu pochodzą z **Databricks Marketplace** — wbudowanego katalogu gotowych datasetów:
# MAGIC 1. Otwórz **Marketplace** w panelu bocznym Databricks
# MAGIC 2. Wyszukaj `Databricks Simulated Retail Customer Data`
# MAGIC 3. Kliknij **Get instant access** — dataset pojawia się jako nowy katalog w Unity Catalog
# MAGIC
# MAGIC Po pobraniu mamy katalog `databricks_simulated_retail_customer_data` ze schematem `v01` i trzema tabelami:
# MAGIC - **customers** — dane klientów B2B (customer_id, name, state, loyalty_segment, units_purchased, lat/lon)
# MAGIC - **sales_orders** — zamówienia (order_number, customer_id, order_datetime, number_of_line_items, ordered_products jako JSON)
# MAGIC - **sales** — produkty (product_id, product_name, category, price)
# MAGIC
# MAGIC ### Czym jest eksploracja danych?
# MAGIC Eksploracja (EDA) to pierwszy krok każdego projektu analitycznego. Zanim budujemy modele, musimy **zrozumieć dane**: ile ich mamy, jaki mają zakres, czy są kompletne.

# COMMAND ----------

# DBTITLE 1,Sekcja 1b: Przegląd warunków licencji Marketplace
# MAGIC %md
# MAGIC ### Sekcja 1b: Zanim klikniesz „Get instant access” — przegląd warunków licencji
# MAGIC
# MAGIC Dataset z Marketplace to **produkt danych dostawcy**, nie nasze dane. Zanim trafi do pipeline’u, Compliance Officer chce wiedzieć **na jakich warunkach** go używamy.
# MAGIC
# MAGIC W listingu Marketplace sprawdź i **zanotuj** (to zapis do audytu):
# MAGIC
# MAGIC | Co sprawdzić | Gdzie w listingu | Dlaczego |
# MAGIC | --- | --- | --- |
# MAGIC | **Dostawca** i typ produktu | Nagłówek, sekcja *Provider* | Kto odpowiada za jakość danych |
# MAGIC | **Licencja / Terms of use** | Sekcja *Terms* lub *License* | Jakie użycie jest dozwolone (produkcja? redystrybucja?) |
# MAGIC | **Polityka prywatności** | Link w warunkach | Czy dane zawierają PII? Jakie regulacje? |
# MAGIC | **Częstotliwość aktualizacji** | Opis lub metadata | Czy dane są jednorazowe czy odświeżane? |
# MAGIC | **Scope danych** | Opis datasetu | Ile wierszy, jaki okres, jakie tabele |
# MAGIC
# MAGIC > **Ćwiczenie:** Otwórz Marketplace, znajdź listing i wypełnij tabelę powyżej.

# COMMAND ----------

# DBTITLE 1,Zadanie 5.2b: Schemat tabeli Gold
# MAGIC %sql
# MAGIC -- ZADANIE 5.2b: Schemat tabeli Gold — sprawdź kolumny PII
# MAGIC --
# MAGIC -- Zwróć uwagę, które kolumny zawierają dane osobowe (PII)!
# MAGIC -- W Warsztacie 2 zabezpieczymy je guardrails.
# MAGIC --
# MAGIC -- a) Schemat tabeli:
# MAGIC --    DESCRIBE TABLE <YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360
# MAGIC --
# MAGIC -- b) Odczytaj dane z wersji 0 (pierwsza wersja po zapisie):
# MAGIC --    SELECT * FROM <YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360 VERSION AS OF 0 LIMIT 5
# MAGIC --
# MAGIC -- Kolumny PII: customer_name, tax_id, lat, lon
# MAGIC -- Kolumny bezpieczne: loyalty_segment, monetary, recency_days, frequency, num_orders
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Zadanie 8b.2: AutoML — porównanie z ręcznym modelem
# ZADANIE 8b.2: Załaduj najlepszy model AutoML i porównaj z ręcznym
#
# Krok 1: Załaduj model AutoML:
#   best_model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"
#   automl_model = mlflow.pyfunc.load_model(best_model_uri)
#
# Krok 2: Predykcja na X_test (te same dane co w Sekcji 6):
#   automl_predictions = automl_model.predict(X_test)
#
# Krok 3: Oblicz metryki i porównaj:
#   from sklearn.metrics import accuracy_score, f1_score
#   acc_automl = accuracy_score(y_test, automl_predictions)
#   f1_automl = f1_score(y_test, automl_predictions, average='weighted')
#   print(f"AutoML   — Accuracy: {acc_automl:.4f}, F1: {f1_automl:.4f}")
#   print(f"Ręczny GB — Accuracy: {acc_gb:.4f}, F1: {f1_gb:.4f}")
#   print(f"Ręczny RF — Accuracy: {acc_rf:.4f}, F1: {f1_rf:.4f}")
#
# Hint: Który model wygrał? Sprawdź F1 — wyższy = lepszy
# Hint: AutoML też loguje wyniki do MLflow Experiment — sprawdź panel boczny



# COMMAND ----------

# DBTITLE 1,Zadanie 1.1: Metadane katalogu Marketplace
# MAGIC %sql
# MAGIC -- ZADANIE 1.1: Zbadaj metadane katalogu z Marketplace
# MAGIC --
# MAGIC -- Katalog z Marketplace to katalog typu Delta Sharing —
# MAGIC -- Unity Catalog przechowuje nazwę dostawcy i share'a.
# MAGIC --
# MAGIC -- Napisz zapytanie, które pokaże metadane katalogu:
# MAGIC --   DESCRIBE CATALOG EXTENDED databricks_simulated_retail_customer_data
# MAGIC --
# MAGIC -- Zwróć uwagę na pola: Catalog Type, Provider Name, Share Name, Owner, Created.
# MAGIC --
# MAGIC -- Następnie wyświetl listę tabel udostępnionych w tym katalogu:
# MAGIC --   SHOW TABLES IN databricks_simulated_retail_customer_data.v01
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Zadanie 1.2: Przegląd danych — rozmiar tabel
# MAGIC %sql
# MAGIC -- ZADANIE 1.2: Ile mamy danych? Jaki zakres?
# MAGIC --
# MAGIC -- Napisz zapytanie UNION ALL, które pokaże dla każdej z 3 tabel:
# MAGIC --   - Nazwę tabeli (stała tekstowa)
# MAGIC --   - Liczbę wierszy (COUNT(*))
# MAGIC --   - Liczbę unikalnych identyfikatorów (COUNT(DISTINCT ...))
# MAGIC --
# MAGIC -- Tabele:
# MAGIC --   databricks_simulated_retail_customer_data.v01.customers   → customer_id
# MAGIC --   databricks_simulated_retail_customer_data.v01.sales_orders → customer_id
# MAGIC --   databricks_simulated_retail_customer_data.v01.sales        → product_id
# MAGIC --
# MAGIC -- Hint: SELECT 'customers' AS tabela, COUNT(*) AS wiersze, COUNT(DISTINCT customer_id) AS unikalne_id
# MAGIC --       FROM ... UNION ALL SELECT 'sales_orders', ...
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Zadanie 1.3: Eksploracja klientów — segmenty i geografia
# MAGIC %sql
# MAGIC -- ZADANIE 1.3: Rozkład klientów per segment lojalności i stan
# MAGIC --
# MAGIC -- loyalty_segment: 0 = nowy, 1 = okazjonalny, 2 = regularny, 3 = VIP
# MAGIC --
# MAGIC -- Napisz zapytanie GROUP BY, które pokaże:
# MAGIC --   - loyalty_segment, state
# MAGIC --   - COUNT(*) AS liczba_klientow
# MAGIC --   - ROUND(AVG(units_purchased), 1) AS avg_units_purchased
# MAGIC --   - ROUND(AVG(lat), 4) AS avg_lat
# MAGIC -- Z tabeli: databricks_simulated_retail_customer_data.v01.customers
# MAGIC -- Posortuj po loyalty_segment, liczba_klientow DESC
# MAGIC --
# MAGIC -- Po uruchomieniu: kliknij "+" przy wyniku → wybierz wykres słupkowy
# MAGIC -- (X = state, Y = liczba_klientow, Color = loyalty_segment)
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Sekcja 2 — Zaawansowana analityka SQL
# MAGIC %md
# MAGIC ## Sekcja 2: Zaawansowana analityka SQL
# MAGIC **Funkcjonalności:** CTE (Common Table Expressions), Window Functions, LAG, RANK, analiza kohortowa
# MAGIC
# MAGIC Teraz przechodzimy do bardziej zaawansowanych zapytań — łączymy tabele klientów z zamówieniami, obliczamy trendy tygodniowe i budujemy ranking klientów.
# MAGIC
# MAGIC ### Kluczowe koncepcje:
# MAGIC - **CTE (Common Table Expression)** — `WITH ... AS (...)` — "tymczasowa tabela" w zapytaniu. Zamiast gnieźddżić podzapytania, rozbijasz logikę na czytelne kroki.
# MAGIC - **Window Functions** — funkcje operujące na "oknie" wierszy powiązanych z bieżącym wierszem:
# MAGIC   - `LAG(col, n)` — wartość z N wierszy wstecz (np. sprzedaż z poprzedniego tygodnia)
# MAGIC   - `RANK() OVER (PARTITION BY ... ORDER BY ...)` — ranking w ramach grupy
# MAGIC - **TRY_DIVIDE** — bezpieczne dzielenie (zwraca NULL zamiast błędu przy dzieleniu przez 0)

# COMMAND ----------

# DBTITLE 1,Zadanie 2.1: Trend tygodniowy zamówień z growth rate
# MAGIC %sql
# MAGIC -- ZADANIE 2.1: Trend tygodniowy zamówień z growth rate (Week-over-Week)
# MAGIC --
# MAGIC -- Odpowiadamy na pytanie: "Jak zmienia się liczba zamówień z tygodnia na tydzień?"
# MAGIC --
# MAGIC -- Napisz zapytanie z CTE i Window Functions:
# MAGIC --
# MAGIC -- CTE 1 (weekly_orders): Agreguj zamówienia tygodniowo
# MAGIC --   - DATE_TRUNC('week', from_unixtime(order_datetime)) AS week
# MAGIC --   - COUNT(*) AS total_orders
# MAGIC --   - SUM(number_of_line_items) AS total_items
# MAGIC --   - COUNT(DISTINCT customer_id) AS unique_customers
# MAGIC -- Źródło: databricks_simulated_retail_customer_data.v01.sales_orders
# MAGIC --
# MAGIC -- CTE 2 (with_growth): Dodaj kolumny z Window Functions
# MAGIC --   - LAG(total_orders, 1) OVER (ORDER BY week) AS prev_week_orders
# MAGIC --   - ROUND(TRY_DIVIDE(total_orders - prev, prev) * 100, 1) AS wow_growth_pct
# MAGIC --
# MAGIC -- Zapytanie końcowe: SELECT * FROM with_growth ORDER BY week
# MAGIC --
# MAGIC -- Hint: TRY_DIVIDE zamiast zwykłego dzielenia — nie wyrzuci błędu gdy prev = 0
# MAGIC -- Po uruchomieniu: dodaj wykres liniowy (X = week, Y = total_orders)
# MAGIC --
# MAGIC -- TO JEST PRZYKŁAD — jeśli chcesz, zrób własną analizę!
# MAGIC -- Ważne jest użycie CTE + LAG/TRY_DIVIDE, nie konkretna metryka. Alternatywy:
# MAGIC --   • Trend miesięczny zamiast tygodniowego (DATE_TRUNC('month', ...))
# MAGIC --   • Growth rate wartości zamówień zamiast liczby
# MAGIC --   • Trend per segment lojalnosci (PARTITION BY loyalty_segment)
# MAGIC --   • Analiza dnia tygodnia: które dni mają najwięcej zamówień?
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Zadanie 2.2: Ranking klientów per segment
# MAGIC %sql
# MAGIC -- ZADANIE 2.2: Top 3 klienci w każdym segmencie lojalności
# MAGIC --
# MAGIC -- Ranking pozwala odpowiedzieć: "Którzy klienci są najlepsi w każdym segmencie?"
# MAGIC -- Używamy RANK() OVER (PARTITION BY ... ORDER BY ...)
# MAGIC --
# MAGIC -- Napisz zapytanie z CTE i podzapytaniem:
# MAGIC --
# MAGIC -- CTE (customer_orders): JOIN customers + sales_orders
# MAGIC --   - c.customer_id, c.customer_name, c.state, c.loyalty_segment
# MAGIC --   - COUNT(o.order_number) AS total_orders
# MAGIC --   - SUM(o.number_of_line_items) AS total_items
# MAGIC --   GROUP BY wszystkie pola klienta
# MAGIC --
# MAGIC -- JOIN: INNER JOIN na customer_id
# MAGIC -- Tabele:
# MAGIC --   databricks_simulated_retail_customer_data.v01.customers c
# MAGIC --   databricks_simulated_retail_customer_data.v01.sales_orders o
# MAGIC --
# MAGIC -- Podzapytanie: Dodaj RANK() OVER (PARTITION BY loyalty_segment ORDER BY total_orders DESC)
# MAGIC -- Filtruj: WHERE rank_in_segment <= 3
# MAGIC -- Sortuj: ORDER BY loyalty_segment, rank_in_segment
# MAGIC --
# MAGIC -- TO JEST PRZYKŁAD — ważne jest użycie RANK() + PARTITION BY. Alternatywy:
# MAGIC --   • Top 3 klienci per STAN (PARTITION BY state) zamiast per segment
# MAGIC --   • Ranking wg wartości monetary zamiast liczby zamówień
# MAGIC --   • Top 5 zamiast top 3 (WHERE rank_in_segment <= 5)
# MAGIC --   • Dodaj DENSE_RANK i porównaj różnice z RANK
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Sekcja 3 — AI Functions
# MAGIC %md
# MAGIC ## Sekcja 3: AI Functions — sztuczna inteligencja w SQL
# MAGIC **Funkcjonalności:** `ai_query()`, `ai_classify()` — LLM bezpośrednio w zapytaniach SQL
# MAGIC
# MAGIC To jedna z najciekawszych możliwości Databricks — wywołujesz model językowy (LLM) jako zwykłą funkcję SQL! Nie potrzebujesz Pythona, API keys, ani infrastruktury.
# MAGIC
# MAGIC ### Dwie główne funkcje:
# MAGIC - **`ai_query(endpoint, prompt)`** — wysyła tekst do LLM i zwraca odpowiedź. Używasz jak każdej innej funkcji SQL (SUM, AVG, CONCAT...), ale dostaje się do LLM.
# MAGIC - **`ai_classify(tekst, ARRAY('kat1', 'kat2', ...))`** — AI automatycznie przypisuje etykietę z podanej listy kategorii.
# MAGIC
# MAGIC ### Ważna optymalizacja:
# MAGIC AI Functions są kosztowne (każde wywołanie = request do LLM). Dlatego **najpierw agregujemy** dane, a dopiero potem wywołujemy AI na małym zbiorze wyników. Nigdy nie wywołuj AI na 28 000 wierszach!

# COMMAND ----------

# DBTITLE 1,Zadanie 3.1: AI generuje insight o segmentach klientów
# MAGIC %sql
# MAGIC -- ZADANIE 3.1: AI generuje biznesowy insight o segmentach klientów
# MAGIC --
# MAGIC -- Cel: Dla każdego segmentu lojalności LLM analizuje statystyki
# MAGIC -- i generuje krótką rekomendację biznesową (1–2 zdania).
# MAGIC --
# MAGIC -- Napisz zapytanie z CTE:
# MAGIC --
# MAGIC -- CTE (segment_stats): Agreguj dane per loyalty_segment
# MAGIC --   - COUNT(*) AS customers
# MAGIC --   - ROUND(AVG(units_purchased), 1) AS avg_purchases
# MAGIC --   - ROUND(AVG(units_purchased) * COUNT(*), 0) AS estimated_total_units
# MAGIC -- Źródło: databricks_simulated_retail_customer_data.v01.customers
# MAGIC --
# MAGIC -- SELECT końcowy: loyalty_segment, customers, avg_purchases, plus:
# MAGIC --   ai_query(
# MAGIC --     'databricks-meta-llama-3-3-70b-instruct',
# MAGIC --     CONCAT('Jesteś analitykiem retail. Segment ', CAST(loyalty_segment AS STRING),
# MAGIC --            ' ma ', CAST(customers AS STRING), ' klientów, ',
# MAGIC --            'średnia zakupów: ', CAST(avg_purchases AS STRING), '. ',
# MAGIC --            'Napisz 1–2 zdania rekomendacji biznesowej po polsku.')
# MAGIC --   ) AS ai_recommendation
# MAGIC --
# MAGIC -- Hint: ai_query zwraca STRING — można go użyć w SELECT jak każdą inną kolumnę
# MAGIC --
# MAGIC -- TO JEST PRZYKŁAD — ważne jest użycie ai_query z CTE. Alternatywy:
# MAGIC --   • Rekomendacje per STAN (top 5 stanów) zamiast per segment
# MAGIC --   • Poproś LLM o analizę ryzyka churnu zamiast rekomendacji
# MAGIC --   • Generuj 3-punktową strategię marketingową per segment
# MAGIC --   • Zmień język promptu na angielski i porównaj wyniki
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Zadanie 3.2: AI klasyfikuje typ klienta
# MAGIC %sql
# MAGIC -- ZADANIE 3.2: AI klasyfikuje typ klienta na podstawie zachowań
# MAGIC --
# MAGIC -- Cel: AI automatycznie przypisuje etykietę strategii obsługi
# MAGIC -- na podstawie danych klienta (bez ręcznego definiowania progów).
# MAGIC --
# MAGIC -- Napisz zapytanie z CTE:
# MAGIC --
# MAGIC -- CTE (top_customers): Pobierz top 10 klientów z NY i CA
# MAGIC --   - c.customer_name, c.state, c.units_purchased, c.loyalty_segment
# MAGIC --   - COUNT(o.order_number) AS total_orders
# MAGIC --   LEFT JOIN sales_orders o ON customer_id
# MAGIC --   WHERE c.state IN ('NY', 'CA')
# MAGIC --   GROUP BY ..., ORDER BY units_purchased DESC, LIMIT 10
# MAGIC --
# MAGIC -- SELECT końcowy: Dodaj kolumnę ai_classify:
# MAGIC --   ai_classify(
# MAGIC --     CONCAT('Customer: ', customer_name, ', zakupy: ', units_purchased,
# MAGIC --            ' szt., zamówienia: ', total_orders, ', segment: ', loyalty_segment),
# MAGIC --     ARRAY('VIP Premium', 'Regularny Aktywny', 'Rozwijający się', 'Nowy/Do Aktywacji')
# MAGIC --   ) AS ai_customer_type
# MAGIC --
# MAGIC -- Hint: ai_classify zwraca jeden z elementów ARRAY — najlepiej pasujący
# MAGIC -- Hint: LIMIT 10 w CTE — optymalizacja kosztów AI
# MAGIC --
# MAGIC -- TO JEST PRZYKŁAD — ważne jest użycie ai_classify z ARRAY kategorii. Alternatywy:
# MAGIC --   • Własne kategorie: ARRAY('Ryzyko churnu', 'Stabilny', 'Rosnący', 'Nowy')
# MAGIC --   • Klasyfikuj klientów z innych stanów (TX, FL, IL)
# MAGIC --   • Klasyfikuj na podstawie RFM zamiast zakupów
# MAGIC --   • Dodaj więcej kategorii (5-6) i sprawdź czy AI radzi sobie z rozróżnieniem
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Sekcja 4 — PySpark: Cleaning, Join i Feature Engineering
# MAGIC %md
# MAGIC ## Sekcja 4: PySpark — Cleaning, Join i Feature Engineering
# MAGIC **Funkcjonalności:** DataFrame API, JSON parsing, JOIN, Window Functions, wielojęzyczność notebooka
# MAGIC
# MAGIC Przechodzimy na Python! W tej sekcji:
# MAGIC 1. **Łączymy tabele** — customers + sales_orders (JOIN)
# MAGIC 2. **Parsujemy JSON** — wyciągamy dane o produktach i promocjach z kolumny `ordered_products` (STRING z JSON array)
# MAGIC 3. **Budujemy cechy klienta** — RFM (Recency, Frequency, Monetary), cechy geograficzne
# MAGIC 4. **Przygotowujemy dane do ML** — predykcja `loyalty_segment`
# MAGIC
# MAGIC ### Czym jest RFM?
# MAGIC - **Recency** — ile dni minęło od ostatniego zamówienia (im mniej, tym lepiej)
# MAGIC - **Frequency** — ile razy klient kupował (więcej = bardziej lojalny)
# MAGIC - **Monetary** — ile łącznie wydał (wartość klienta)
# MAGIC
# MAGIC > **Wielojęzyczność:** Zwróć uwagę, że SQL używaliśmy wyżej, a teraz przechodzimy na Python w tym samym notebooku.

# COMMAND ----------

# DBTITLE 1,Zadanie 4.1: Feature Engineering — customer_360
# ZADANIE 4.1: Feature Engineering — zbuduj tabelę customer_360
#
# Cel: Z tabel customers i sales_orders stworzysz jedną tabelę Gold
# z pełnym profilem każdego klienta (cechy RFM + statystyki zamówień).
#
# Krok 1: Importy i wczytanie danych
#   from pyspark.sql import functions as F
#   from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
#   catalog = "databricks_simulated_retail_customer_data"
#   customers = spark.table(f"{catalog}.v01.customers")
#   orders = spark.table(f"{catalog}.v01.sales_orders")
#
# Krok 2: Parsowanie JSON z ordered_products
#   Kolumna ordered_products to string z JSON array — wyciągnij łączną wartość:
#   schema = ArrayType(StructType([StructField("id", StringType()), ...]))
#   orders_parsed = orders.withColumn("products", from_json("ordered_products", schema))
#   orders_value = orders_parsed.withColumn("order_value",
#       F.aggregate("products", F.lit(0.0), lambda acc, x: acc + x["qty"] * x["price"]))
#
# Krok 3: Agregacja per klient (RFM + statystyki)
#   order_stats = orders_value.groupBy("customer_id").agg(
#       F.count("*").alias("num_orders"),
#       F.sum("order_value").alias("monetary"),
#       F.avg("order_value").alias("avg_item_value"),
#       F.min(F.from_unixtime("order_datetime")).alias("first_order_date"),
#       F.max(F.from_unixtime("order_datetime")).alias("last_order_date"),
#       F.sum("number_of_line_items").alias("frequency"),
#       F.sum(F.when(F.col("promo_info").isNotNull(), 1).otherwise(0)).alias("promo_orders")
#   )
#
# Krok 4: JOIN z customers (LEFT JOIN — nie każdy klient ma zamówienia)
#   gold_df = customers.join(order_stats, "customer_id", "left")
#
# Krok 5: Dodaj cechy obliczane:
#   - recency_days = datediff(current_date(), last_order_date)
#   - has_orders = CASE WHEN num_orders > 0 THEN true ELSE false
#   - promo_ratio = promo_orders / num_orders
#   Wypełnij nulle (fillna) dla klientów bez zamówień
#
# Krok 6: display(gold_df.limit(10))
#
# Hint: from_json wymaga zdefiniowania schematu JSON
# Hint: F.aggregate to wbudowana funkcja PySpark do redukcji tablic
# Hint: fillna({"num_orders": 0, "monetary": 0.0, ...}) dla nullów



# COMMAND ----------

# DBTITLE 1,Sekcja 5 — Delta Lake
# MAGIC %md
# MAGIC ## Sekcja 5: Delta Lake — zapis i wersjonowanie
# MAGIC **Funkcjonalności:** Delta Lake, saveAsTable, Time Travel, DESCRIBE HISTORY
# MAGIC
# MAGIC Zapisujemy przetworzoną tabelę `gold_customer_360` jako tabelę Delta w Unity Catalog. To będzie nasza **główna tabela** — używana w ML, dashboardach, i w następnych warsztatach (guardrails, RAG, agent).
# MAGIC
# MAGIC Delta Lake daje nam:
# MAGIC - **ACID transactions** — niezawodność zapisu (albo się uda cały, albo wcale)
# MAGIC - **Time Travel** — dostęp do poprzednich wersji danych
# MAGIC - **Schema enforcement** — ochrona przed błędnymi danymi
# MAGIC
# MAGIC > **Ważne:** Ta tabela zawiera dane PII klientów (customer_name, tax_id, adresy) — w Warsztacie 2 zabezpieczymy ją guardrails.

# COMMAND ----------

# DBTITLE 1,Zadanie 5.1: Zapis tabeli Delta (Gold)
# ZADANIE 5.1: Zapisz DataFrame gold_df jako tabelę Delta w Unity Catalog
#
# Krok 1: Zdefiniuj nazwę tabeli Gold:
#   GOLD_TABLE = "<YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360"
#
# Krok 2: Zapisz dane:
#   gold_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_TABLE)
#
# Krok 3: Zweryfikuj zapis:
#   print(f"Tabela zapisana: {GOLD_TABLE}")
#   print(f"  Wiersze: {spark.table(GOLD_TABLE).count():,}")
#   print(f"  Kolumny: {len(spark.table(GOLD_TABLE).columns)}")
#
# Hint: mode("overwrite") nadpisze istniejącą tabelę (jeśli już istnieje)



# COMMAND ----------

# DBTITLE 1,Zadanie 5.2: Time Travel
# MAGIC %sql
# MAGIC -- ZADANIE 5.2: Time Travel — historia zmian tabeli
# MAGIC --
# MAGIC -- Każdy zapis tworzy nową wersję — możemy wrócić do dowolnej!
# MAGIC --
# MAGIC -- a) Pokaż historię zmian tabeli:
# MAGIC --    DESCRIBE HISTORY <YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360
# MAGIC --
# MAGIC -- b) Sprawdź schemat tabeli Gold (zwróć uwagę na kolumny PII!):
# MAGIC --    DESCRIBE TABLE <YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360
# MAGIC --
# MAGIC -- c) (Opcjonalnie) Odczytaj dane z poprzedniej wersji:
# MAGIC --    SELECT * FROM <YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360 VERSION AS OF 0 LIMIT 5
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Sekcja 5b — Tool Calling
# MAGIC %md
# MAGIC ## Sekcja 5b: Tool Calling — UC Function jako narzędzie LLM
# MAGIC **Funkcjonalności:** UC Functions, Foundation Model API, tool calling (function calling)
# MAGIC
# MAGIC W sekcji 3 widzieliśmy `ai_query()` — LLM odpowiada na pytanie **w SQL**. Ale co jeśli chcemy, żeby **LLM sam zdecydował jaką funkcję wywołać**? To jest **tool calling**:
# MAGIC
# MAGIC ```
# MAGIC Użytkownik: „Jaki jest łączny przychód od VIPów w NY?”
# MAGIC    ↓
# MAGIC LLM decyduje: "Muszę wywołać get_revenue_summary(segment=3, state='NY')"
# MAGIC    ↓
# MAGIC UC Function wykonuje SQL na tabeli Gold i zwraca wynik
# MAGIC    ↓
# MAGIC LLM formatuje odpowiedź dla użytkownika
# MAGIC ```
# MAGIC
# MAGIC To fundament **agentów AI** — w Warsztacie 4 zbudujemy pełnego agenta z wieloma narzędziami.

# COMMAND ----------

# DBTITLE 1,Zadanie 5b.1: UC Function
# MAGIC %sql
# MAGIC -- ZADANIE 5b.1: Utwórz UC Function zwracającą podsumowanie przychodu
# MAGIC --
# MAGIC -- Ta funkcja będzie narzędziem LLM — agent wywoła ją automatycznie.
# MAGIC --
# MAGIC -- CREATE OR REPLACE FUNCTION <YOUR_CATALOG>.<YOUR_SCHEMA>.get_revenue_summary(
# MAGIC --   segment BIGINT COMMENT 'Loyalty segment ID: 0–3. Pass -1 for all.',
# MAGIC --   state_filter STRING COMMENT 'US state code (e.g. NY). Pass ALL for all states.'
# MAGIC -- )
# MAGIC -- RETURNS STRING
# MAGIC -- COMMENT 'Returns total revenue, customer count and avg revenue per segment/state.'
# MAGIC -- RETURN SELECT CONCAT_WS('\n',
# MAGIC --   CONCAT('Segment: ', COALESCE(CAST(... AS STRING), 'ALL')),
# MAGIC --   CONCAT('State: ', ...),
# MAGIC --   CONCAT('Customers: ', COUNT(*)),
# MAGIC --   CONCAT('Total Revenue: $', ROUND(SUM(monetary), 2)),
# MAGIC --   CONCAT('Avg Revenue: $', ROUND(AVG(monetary), 2))
# MAGIC -- )
# MAGIC -- FROM <YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360
# MAGIC -- WHERE (segment = -1 OR loyalty_segment = segment)
# MAGIC --   AND (state_filter = 'ALL' OR state = state_filter)
# MAGIC --
# MAGIC -- Po utworzeniu przetestuj: SELECT <YOUR_CATALOG>.<YOUR_SCHEMA>.get_revenue_summary(3, 'NY')
# MAGIC --
# MAGIC -- TO JEST PRZYKŁAD — ważne jest stworzenie UC Function z COMMENT i parametrami. Alternatywy:
# MAGIC --   • Funkcja `get_churn_risk(segment)` — zwraca avg recency_days i % bez zamówień
# MAGIC --   • Funkcja `get_state_summary(state)` — liczba klientów, segmenty, avg monetary per stan
# MAGIC --   • Funkcja `compare_segments(seg_a, seg_b)` — porównanie dwóch segmentów w jednym wywołaniu
# MAGIC --   • Dodaj trzeci parametr np. `min_monetary` filtrujący klientów poniżej progu
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Zadanie 5b.2: Tool Calling z LLM
# ZADANIE 5b.2: Tool calling — LLM sam wywołuje UC Function
#
# Krok 1: Setup klienta OpenAI (Foundation Model API)
#   from openai import OpenAI
#   from databricks.sdk import WorkspaceClient
#   import json
#   w = WorkspaceClient()
#   client = OpenAI(
#       api_key=w.config.authenticate()["Authorization"].split(" ", 1)[1],
#       base_url=f"{w.config.host}/serving-endpoints"
#   )
#
# Krok 2: Zdefiniuj narzędzie (tool) opisujące UC Function:
#   tools = [{
#     "type": "function",
#     "function": {
#       "name": "<YOUR_CATALOG>__<YOUR_SCHEMA>__get_revenue_summary",
#       "description": "Returns total revenue per segment and state",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "segment": {"type": "integer", "description": "Loyalty segment 0–3, -1 for all"},
#           "state_filter": {"type": "string", "description": "US state code or ALL"}
#         }, "required": ["segment", "state_filter"]
#       }
#     }
#   }]
#
# Krok 3: Wyślij pytanie z narzędziami:
#   response = client.chat.completions.create(
#     model="databricks-meta-llama-3-3-70b-instruct",
#     messages=[{"role": "user", "content": "Jaki jest przychód od VIPów w Nowym Jorku?"}],
#     tools=tools
#   )
#
# Krok 4: Jeśli LLM wywołał narzędzie — wykonaj funkcję i zwróć wynik:
#   tool_call = response.choices[0].message.tool_calls[0]
#   args = json.loads(tool_call.function.arguments)
#   result = spark.sql(f"SELECT <YOUR_CATALOG>.<YOUR_SCHEMA>.get_revenue_summary({args['segment']}, '{args['state_filter']}')").collect()[0][0]
#   print(f"LLM chce wywołać: {tool_call.function.name}({args})")
#   print(f"Wynik: {result}")
#
# TO JEST PRZYKŁAD — ważne jest zrozumienie pętli tool calling. Alternatywy:
#   • Zmień pytanie: „Jaka jest średnia wartość klienta w Kalifornii?” i sprawdź args
#   • Zadaj pytanie wymagające dwóch wywołań: „Porównaj przychód VIP w NY vs CA”
#   • Zadaj pytanie które NIE pasuje do narzędzia i sprawdź co LLM zrobi
#   • Dodaj drugie narzędzie (np. get_customer_profile) i testuj routing



# COMMAND ----------

# DBTITLE 1,Sekcja 6 — Machine Learning z MLflow
# MAGIC %md
# MAGIC ## Sekcja 6: Machine Learning z MLflow
# MAGIC **Funkcjonalności:** scikit-learn, MLflow autologging, tracking eksperymentów, porównanie runów
# MAGIC
# MAGIC MLflow to open-source platforma do zarządzania cyklem życia modeli ML. W Databricks jest w pełni zintegrowana:
# MAGIC - **Autologging** — `mlflow.sklearn.autolog()` automatycznie zapisuje parametry, metryki, model
# MAGIC - **Tracking** — każdy trening ("run") jest zapisany, możesz porównać setki eksperymentów w UI
# MAGIC - **Model Registry** — najlepszy model rejestrujesz jako wersję w Unity Catalog
# MAGIC
# MAGIC ### Nasz problem ML:
# MAGIC **Klasyfikacja loyalty_segment** — na podstawie cech klienta (RFM, geografia, zamówienia) przewidujemy, do którego segmentu lojalności należy klient (0–3).
# MAGIC
# MAGIC ### Metryki:
# MAGIC - **Accuracy** — % poprawnych predykcji (prosta, ale może być myląca przy niezbalansowanych klasach)
# MAGIC - **F1-score (weighted)** — harmoniczna średnia precyzji i recall, ważona licznością klas
# MAGIC
# MAGIC > **Plan:** Trenujemy 2 modele (Gradient Boosting i Random Forest), porównujemy metryki, najlepszy rejestrujemy w UC.

# COMMAND ----------

# DBTITLE 1,Zadanie 6.1: Przygotowanie danych do modelu
# ZADANIE 6.1: Przygotowanie danych do modelu klasyfikacji
#
# Krok 1: Importy
#   import mlflow, mlflow.sklearn
#   from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
#   from sklearn.model_selection import train_test_split
#   from sklearn.metrics import accuracy_score, f1_score, classification_report
#   import pandas as pd, numpy as np
#
# Krok 2: Wczytaj dane Gold i przekonwertuj na pandas
#   GOLD_TABLE = "<YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360"
#   feature_cols = ["units_purchased", "recency_days", "frequency", "num_orders",
#                   "monetary", "avg_item_value", "promo_orders", "promo_ratio"]
#   target_col = "loyalty_segment"
#
#   pdf = (spark.table(GOLD_TABLE)
#       .select(feature_cols + [target_col])
#       .dropna()
#       .toPandas())
#
# Krok 3: Podział na X (features) i y (target)
#   X = pdf[feature_cols]
#   y = pdf[target_col]
#
# Krok 4: train_test_split
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#   print(f"Train: {len(X_train)}, Test: {len(X_test)}, Klasy: {sorted(y.unique())}")



# COMMAND ----------

# DBTITLE 1,Zadanie 6.2: Model 1 — Gradient Boosting
# ZADANIE 6.2: Trening modelu Gradient Boosting Classifier z MLflow
#
# Krok 1: Włącz autologging
#   mlflow.sklearn.autolog()
#
# Krok 2: Trenuj model w kontekście MLflow run:
#   with mlflow.start_run(run_name="GBClassifier_loyalty"):
#       model_gb = GradientBoostingClassifier(
#           n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
#       )
#       model_gb.fit(X_train, y_train)
#       preds_gb = model_gb.predict(X_test)
#
#       acc = accuracy_score(y_test, preds_gb)
#       f1 = f1_score(y_test, preds_gb, average='weighted')
#       print(f"Gradient Boosting — Accuracy: {acc:.4f}, F1: {f1:.4f}")
#       print(classification_report(y_test, preds_gb))
#
# Hint: Po uruchomieniu sprawdź MLflow Experiment w panelu bocznym
# Hint: autolog() automatycznie loguje parametry, metryki i model



# COMMAND ----------

# DBTITLE 1,Zadanie 6.3: Model 2 — Random Forest
# ZADANIE 6.3: Trening modelu Random Forest Classifier (do porównania)
#
# Ten sam schemat co 6.2, ale inny algorytm:
#   with mlflow.start_run(run_name="RFClassifier_loyalty"):
#       model_rf = RandomForestClassifier(
#           n_estimators=200, max_depth=10, random_state=42
#       )
#       model_rf.fit(X_train, y_train)
#       preds_rf = model_rf.predict(X_test)
#
#       acc = accuracy_score(y_test, preds_rf)
#       f1 = f1_score(y_test, preds_rf, average='weighted')
#       print(f"Random Forest — Accuracy: {acc:.4f}, F1: {f1:.4f}")
#       print(classification_report(y_test, preds_rf))
#
# Hint: Po uruchomieniu obu modeli sprawdź MLflow Experiment —
#   zobaczysz oba runy obok siebie z metrykami



# COMMAND ----------

# DBTITLE 1,Sekcja 7 — Model Registry
# MAGIC %md
# MAGIC ## Sekcja 7: Model Registry w Unity Catalog
# MAGIC **Funkcjonalność:** Rejestracja modelu, wersjonowanie, governance modeli
# MAGIC
# MAGIC Model Registry to **centralny rejestr modeli ML** — zamiast trzymać model jako plik pickle, rejestrujemy go w Unity Catalog:
# MAGIC - **Wersjonowanie** — każda nowa wersja modelu ma swój numer (v1, v2...)
# MAGIC - **Governance** — uprawnienia, audyt, liniaż danych
# MAGIC - **Współdzielenie** — inny zespół może załadować Twój model jedną linią kodu

# COMMAND ----------

# DBTITLE 1,Zadanie 7.1: Rejestracja modelu w UC
# ZADANIE 7.1: Rejestracja najlepszego modelu w Unity Catalog
#
# Krok 1: Pobierz ostatni run MLflow:
#   best_run = mlflow.last_active_run()
#
# Krok 2: Zarejestruj model:
#   model_name = "<YOUR_CATALOG>.<YOUR_SCHEMA>.loyalty_classifier"
#   registered_model = mlflow.register_model(
#       model_uri=f"runs:/{best_run.info.run_id}/model",
#       name=model_name
#   )
#
# Krok 3: Wypisz informacje:
#   print(f"Model: {model_name}")
#   print(f"Wersja: {registered_model.version}")
#   print(f"Run ID: {best_run.info.run_id}")
#
# Hint: mlflow.last_active_run() zwraca ostatni run (Random Forest z 6.3)



# COMMAND ----------

# DBTITLE 1,Sekcja 8 — Batch Inference
# MAGIC %md
# MAGIC ## Sekcja 8: Predykcja na nowych danych (Batch Inference)
# MAGIC **Funkcjonalność:** Model jako Spark UDF, skalowalna inferencja
# MAGIC
# MAGIC Batch Inference to zastosowanie modelu na **dużym zbiorze danych naraz**:
# MAGIC 1. Ładujemy model z Unity Catalog: `mlflow.pyfunc.load_model("models:/nazwa/latest")`
# MAGIC 2. Model działa jak **zwykła funkcja** — podajesz DataFrame, dostajesz predykcje
# MAGIC 3. Dzięki Spark może przetworzyć miliony wierszy równolegle
# MAGIC
# MAGIC To typowy scenariusz: co noc generujesz predykcje segmentu lojalności dla nowych klientów.

# COMMAND ----------

# DBTITLE 1,Zadanie 8.1: Predykcja loyalty_segment
# ZADANIE 8.1: Załaduj model z UC i wykonaj predykcję
#
# Krok 1: Załaduj model:
#   import os
#   os.environ["MLFLOW_OPENAI_RETRIES"] = "0"
#   model_name = "<YOUR_CATALOG>.<YOUR_SCHEMA>.loyalty_classifier"
#   loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
#
# Krok 2: Przygotuj dane testowe (pandas):
#   test_sample = X_test.head(20)
#
# Krok 3: Wykonaj predykcję:
#   predictions = loaded_model.predict(test_sample)
#
# Krok 4: Porównaj z rzeczywistymi wartościami:
#   comparison = pd.DataFrame({
#       "Actual": y_test.head(20).values,
#       "Predicted": predictions
#   })
#   comparison["Correct"] = comparison["Actual"] == comparison["Predicted"]
#   display(comparison)
#   print(f"Trafność: {comparison['Correct'].mean():.1%}")



# COMMAND ----------

# DBTITLE 1,Sekcja 8b — AutoML
# MAGIC %md
# MAGIC ## Sekcja 8b: AutoML — automatyczny dobór modelu
# MAGIC **Funkcjonalność:** Databricks AutoML, automatyczne porównanie modeli, feature importance
# MAGIC
# MAGIC AutoML to **automatyczny dobieralnik modeli**. Zamiast ręcznie testować różne algorytmy, AutoML robi to za Ciebie:
# MAGIC 1. Testuje wiele algorytmów (XGBoost, LightGBM, RandomForest...)
# MAGIC 2. Optymalizuje hiperparametry
# MAGIC 3. Generuje **gotowy notebook** z najlepszym modelem
# MAGIC 4. Loguje wszystko do MLflow
# MAGIC
# MAGIC > **Wymaga klastra z ML Runtime** (np. 16.x ML). Na Serverless compute AutoML nie jest dostępny.
# MAGIC > Przed uruchomieniem tej sekcji przybij się na klaster z ML Runtime.

# COMMAND ----------

# DBTITLE 1,Zadanie 8b.1: AutoML — klasyfikacja
# ZADANIE 8b.1: Uruchom AutoML (klasyfikacja) na danych Gold
# UWAGA: Wymaga klastra z ML Runtime!
#
# Krok 1: Import AutoML:
#   try:
#       from databricks import automl
#   except ImportError:
#       print("AutoML wymaga klastra z ML Runtime")
#       automl = None
#
# Krok 2: Przygotuj dane (te same features co w Sekcji 6):
#   automl_df = spark.table("<YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360").select(
#       "units_purchased", "recency_days", "frequency", "num_orders",
#       "monetary", "avg_item_value", "promo_orders", "promo_ratio",
#       "loyalty_segment"
#   ).dropna()
#
# Krok 3: Uruchom AutoML:
#   summary = automl.classify(
#       dataset=automl_df,
#       target_col="loyalty_segment",
#       primary_metric="f1",
#       timeout_minutes=5,
#       max_trials=10
#   )
#
# Krok 4: Wypisz najlepszy model:
#   print(summary.best_trial.model_description)
#   print(summary.best_trial.metrics)



# COMMAND ----------

# DBTITLE 1,Sekcja 9 — Prognoza przychodów
# MAGIC %md
# MAGIC ## Sekcja 9: Prognoza przychodów (Time Series Forecasting)
# MAGIC **Funkcjonalność:** RandomForestRegressor, cechy czasowe, prognoza szeregów czasowych
# MAGIC
# MAGIC Prognoza (forecasting) to **przewidywanie przyszłych wartości** na podstawie historycznych danych. Tutaj prognozujemy dzienny przychód na 30 dni do przodu.
# MAGIC
# MAGIC ### Jak to robimy?
# MAGIC Używamy **Random Forest** trenowanego na **cechach czasowych** wyciągniętych z daty zamówień:
# MAGIC - Dzień tygodnia, miesiąc, tydzień roku, weekend (0/1), trend
# MAGIC
# MAGIC ### Przedziały ufności
# MAGIC Random Forest składa się z wielu drzew decyzyjnych. Rozrzut ich predykcji to naturalna miara **niepewności**. Używamy percentyli (2.5% i 97.5%) jako granic prognozy.

# COMMAND ----------

# DBTITLE 1,Zadanie 9.1: Prognoza dziennego przychodu
# ZADANIE 9.1: Prognoza dziennego przychodu (30 dni)
#
# Krok 1: Przygotuj dane czasowe z zamówień:
#   from sklearn.ensemble import RandomForestRegressor
#   import pandas as pd, numpy as np
#
#   orders = spark.table("databricks_simulated_retail_customer_data.v01.sales_orders")
#   daily_revenue = (orders
#       .withColumn("date", F.to_date(F.from_unixtime("order_datetime")))
#       .groupBy("date").agg(F.sum("number_of_line_items").alias("revenue"))
#       .toPandas().sort_values("date"))
#
# Krok 2: Funkcja make_time_features(dates):
#   def make_time_features(dates):
#       df = pd.DataFrame({"date": dates})
#       df["day_of_week"] = df["date"].dt.dayofweek
#       df["month"] = df["date"].dt.month
#       df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
#       df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
#       df["trend"] = range(len(df))
#       return df.drop(columns=["date"])
#
# Krok 3: Trenuj model:
#   X = make_time_features(daily_revenue["date"])
#   y = daily_revenue["revenue"]
#   model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
#   model.fit(X, y)
#
# Krok 4: Prognoza 30 dni:
#   future_dates = pd.date_range(daily_revenue["date"].max() + pd.Timedelta(days=1), periods=30)
#   X_future = make_time_features(future_dates)
#   X_future["trend"] = range(len(X), len(X) + 30)
#   pred = model.predict(X_future)
#
# Krok 5: Przedziały ufności:
#   tree_preds = np.array([t.predict(X_future) for t in model.estimators_])
#   lower = np.percentile(tree_preds, 2.5, axis=0)
#   upper = np.percentile(tree_preds, 97.5, axis=0)
#
# Krok 6: display() wyników jako Spark DataFrame
#
# TO JEST PRZYKŁAD — ważne jest użycie cech czasowych + RF Regressor. Alternatywy:
#   • Prognozuj liczbę zamówień zamiast przychodu
#   • Prognozuj 14 dni zamiast 30 (i porównaj przedziały ufności)
#   • Dodaj cechę „miesiąc” lub „quartał” do modelu i sprawdź feature importance
#   • Spróbuj GradientBoostingRegressor zamiast RF i porównaj R²



# COMMAND ----------

# DBTITLE 1,Sekcja 10 — Dashboard i Genie Space
# MAGIC %md
# MAGIC ## Sekcja 10: Udostępnianie wyników — Dashboard + Genie Space
# MAGIC **Funkcjonalność:** Databricks SDK, Lakeview Dashboard API, Genie Spaces API
# MAGIC
# MAGIC Najlepszy model i analiza są **bezwartościowe**, jeśli wyniki nie dotrą do osób podejmujących decyzje.
# MAGIC
# MAGIC ### Co tworzymy?
# MAGIC - **AI/BI Dashboard** — interaktywny dashboard z wykresami, KPI, tabelami. Tworzymy go **programowo** z kodu.
# MAGIC - **Genie Space** — "chatbot" nad danymi. Użytkownik biznesowy pisze pytanie po polsku, a Genie generuje SQL i zwraca odpowiedź. Nie musi znać SQL!

# COMMAND ----------

# DBTITLE 1,Zadanie 10.1: Tworzenie AI/BI Dashboard (SDK)
# ZADANIE 10.1: Tworzenie AI/BI Dashboard programowo (Lakeview API)
#
# Krok 1: Setup SDK
#   import json, requests
#   from databricks.sdk import WorkspaceClient
#   w = WorkspaceClient()
#   host = w.config.host
#   headers = w.config.authenticate()
#
# Krok 2: Zdefiniuj datasety SQL (lista dictów):
#   datasets = [
#     {"name": "kpi_metrics", "displayName": "KPI",
#      "query": "SELECT COUNT(*) as total_customers, ... FROM <YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360"},
#     {"name": "segment_distribution", ...},
#     {"name": "revenue_by_state", ...}
#   ]
#
# Krok 3: Zdefiniuj widgety (counter, bar, pie):
#   Każdy widget = {"name": ..., "queries": [...], "spec": {"widgetType": ..., "encodings": ...}}
#
# Krok 4: Layout strony i serializacja:
#   serialized = json.dumps({"datasets": datasets, "pages": [{"name": "overview", ...}]})
#
# Krok 5: POST do Lakeview API:
#   resp = requests.post(f"{host}/api/2.0/lakeview/dashboards",
#       headers=headers,
#       json={"display_name": "Customer Intelligence Dashboard", "serialized_dashboard": serialized})
#
# Hint: Spec widgetów: version=2 dla counter, version=3 dla bar/pie
# Hint: Sprawdź notebook źródłowy po szczegóły formatów
#
# TO JEST PRZYKŁAD — ważna jest umiejętność tworzenia dashboardu z kodu. Alternatywy:
#   • Dodaj widget z prognozą z Sekcji 9 (line chart z prediction + confidence)
#   • Dodaj mapę geograficzną (avg monetary per state)
#   • Zmień KPI: zamiast total_customers pokaż churn_rate lub avg_monetary
#   • Dodaj filtr interaktywny per segment lojalnosci



# COMMAND ----------

# DBTITLE 1,Zadanie 10.2: Tworzenie Genie Space (SDK)
# ZADANIE 10.2: Tworzenie Genie Space (REST API)
#
# Genie Space to "chatbot" nad danymi — użytkownik biznesowy pisze pytanie,
# a Genie automatycznie generuje SQL i zwraca odpowiedź.
#
# Krok 1: Pobierz user_email i warehouse_id:
#   user_email = spark.sql("SELECT current_user()").collect()[0][0]
#   wh_response = requests.get(f"{host}/api/2.0/sql/warehouses", headers=headers)
#   warehouse_id = ... (wybierz dostępny warehouse)
#
# Krok 2: Zdefiniuj serialized_space (JSON):
#   - sample_questions: przykładowe pytania po polsku
#     ("Ilu mamy klientów VIP?", "Jaki jest średni przychód per segment?")
#   - data_sources.tables: <YOUR_CATALOG>.<YOUR_SCHEMA>.gold_customer_360
#   - instructions: "Odpowiadaj po polsku."
#
# Krok 3: POST do Genie API:
#   response = requests.post(
#       f"{host}/api/2.0/genie/spaces",
#       headers=headers,
#       json={
#           "title": "Customer Intelligence Assistant",
#           "description": "Asystent do analizy klientów B2B",
#           "serialized_space": serialized_space,
#           "warehouse_id": warehouse_id,
#           "parent_path": f"/Workspace/Users/{user_email}"
#       })
#
# Hint: import uuid; uuid.uuid4().hex dla unikalnych ID
#
# TO JEST PRZYKŁAD — ważne jest stworzenie Genie Space z kodu. Alternatywy:
#   • Napisz własne sample_questions dostosowane do Twojej analizy
#   • Dodaj instrukcje: „Odpowiadaj krótko, max 2 zdania” lub „Dodawaj wykresy”
#   • Podepnij dodatkowe tabele (sales_orders, sales) obok gold_customer_360
#   • Spróbuj zadać pytanie po angielsku i porównać jakość odpowiedzi



# COMMAND ----------

# DBTITLE 1,Podsumowanie workshopu
# MAGIC %md
# MAGIC ## Podsumowanie — co zbudowaliśmy w tym warsztacie
# MAGIC
# MAGIC | # | Sekcja | Funkcjonalność Databricks | Język |
# MAGIC |---|---|---|---|
# MAGIC | 1 | Marketplace i eksploracja | Unity Catalog, SQL, Marketplace, wizualizacje | SQL |
# MAGIC | 2 | Zaawansowana analityka | CTE, Window Functions, LAG, RANK | SQL |
# MAGIC | 3 | AI Functions | ai_query(), ai_classify() — LLM w SQL | SQL |
# MAGIC | 4 | Feature Engineering | PySpark DataFrame API, JSON parsing, JOIN, RFM | Python |
# MAGIC | 5 | Delta Lake | saveAsTable, Time Travel, DESCRIBE HISTORY | Python + SQL |
# MAGIC | 5b | Tool Calling | UC Functions, Foundation Model API | SQL + Python |
# MAGIC | 6 | Machine Learning | scikit-learn, MLflow autologging, klasyfikacja | Python |
# MAGIC | 7 | Model Registry | Rejestracja w Unity Catalog | Python |
# MAGIC | 8 | Batch Inference | Model jako Spark UDF | Python |
# MAGIC | 8b | AutoML | Databricks AutoML, automatyczny dobór modelu | Python (ML Runtime) |
# MAGIC | 9 | Forecasting | Random Forest, cechy czasowe, prognoza | Python |
# MAGIC | 10 | Dashboard + Genie | Lakeview API, Genie Spaces API | Python (SDK) |
# MAGIC
# MAGIC **Następny krok:** Warsztat 2 — Guardrails, Monitoring i Ewaluacja
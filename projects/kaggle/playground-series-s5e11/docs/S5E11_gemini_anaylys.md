

# **Zaawansowane Strategie Predykcji Ryzyka Kredytowego: Wzbogacanie Danych, Inżynieria Cech i Optymalizacja AutoML**

## **Streszczenie Wykonawcze**

Współczesna bankowość i sektor technologii finansowych (FinTech) przechodzą fundamentalną transformację, odchodząc od tradycyjnych, eksperckich kart oceny ryzyka (scorecards) na rzecz zaawansowanych algorytmów uczenia maszynowego. Zdolność do precyzyjnego szacowania prawdopodobieństwa niewypłacalności (Probability of Default \- PD) stanowi o "być albo nie być" instytucji pożyczkowej. Niniejszy raport badawczy stanowi wyczerpującą analizę strategii optymalizacji modeli predykcyjnych w oparciu o referencyjny zbiór danych *Loan Prediction* (Analytics Vidhya/Kaggle). Dokument ten, wykraczając poza standardowe podejścia, integruje wiedzę z zakresu ekonometrii, inżynierii cech oraz zautomatyzowanego uczenia maszynowego (AutoML), ze szczególnym uwzględnieniem biblioteki AutoGluon.  
Analiza wykazuje, że choć oryginalny zestaw danych zawiera silne sygnały predykcyjne – w szczególności historię kredytową – osiągnięcie wyników klasy "state-of-the-art" wymaga wyjścia poza ramy dostarczonej tabeli. Raport szczegółowo omawia konieczność implementacji zewnętrznych danych makroekonomicznych i geoprzestrzennych, transformację zmiennych ciągłych w celu niwelacji skośności rozkładów (transformacje logarytmiczne, Boxa-Coxa) oraz konstrukcję domenowych wskaźników finansowych, takich jak wskaźnik długu do dochodu (DTI) czy wskaźnik pokrycia raty (EMI sufficiency). W dalszej części dokument dokonuje dekonstrukcji zwycięskich rozwiązań z platformy Kaggle, wskazując na kluczową rolę ensemblingu (stacking, bagging) oraz technik radzenia sobie z nierównowagą klas (SMOTE, ADASYN). Finalnie, raport przeprowadza głęboką techniczną dysekcję możliwości optymalizacyjnych frameworku AutoGluon, argumentując za porzuceniem domyślnych ustawień zeroshot na rzecz wielowarstwowego stackingu (best\_quality) oraz manualnej kalibracji przestrzeni hiperparametrów w celu maksymalizacji metryk takich jak ROC-AUC i F1-Score.  
---

## **1\. Wstęp: Paradygmat Ryzyka Kredytowego w Erze Big Data**

### **1.1. Ewolucja Oceny Zdolności Kredytowej**

Tradycyjna ocena zdolności kredytowej opierała się na sztywnych regułach i liniowych modelach dyskryminacyjnych, które, choć interpretowalne, często nie potrafiły uchwycić nieliniowych zależności między cechami demograficznymi a behawioralnymi klienta. Wraz z nadejściem ery Big Data i wzrostem mocy obliczeniowej, instytucje finansowe zaczęły wdrażać algorytmy oparte na drzewach decyzyjnych (Gradient Boosting Machines, Random Forests), które pozwalają na znacznie bardziej granularną segmentację ryzyka.1 Problem, z którym mierzymy się w niniejszym raporcie – klasyfikacja wniosku kredytowego jako "Zatwierdzony" lub "Odrzucony" – jest w istocie problemem estymacji ryzyka defaultu. Zmienna celu Loan\_Status (Y/N) jest binarnym proxy dla złożonego zjawiska wypłacalności finansowej w długim horyzoncie czasowym.2

### **1.2. Charakterystyka Problemu Biznesowego**

Firma "Dream Housing Finance", będąca podmiotem analizy w zbiorze danych, operuje w środowisku o wysokiej zmienności, obsługując klientów z obszarów miejskich, półmiejskich i wiejskich. Automatyzacja procesu decyzyjnego (real-time eligibility check) jest krytyczna dla skalowalności biznesu.1 Błąd pierwszego rodzaju (False Positive – udzielenie kredytu niewypłacalnemu klientowi) skutkuje bezpośrednią stratą kapitału (Loss Given Default). Błąd drugiego rodzaju (False Negative – odrzucenie wiarygodnego klienta) prowadzi do utraty przychodu z odsetek i potencjalnej utraty udziału w rynku. W związku z tym, optymalizacja modelu nie jest jedynie ćwiczeniem akademickim, ale bezpośrednio przekłada się na wynik finansowy (PNL) przedsiębiorstwa. Analiza literatury wskazuje, że modele uczenia maszynowego, takie jak XGBoost czy LightGBM, wykazują wyższość nad tradycyjnymi metodami statystycznymi w minimalizacji obu rodzajów błędów, pod warunkiem odpowiedniego przetworzenia danych wejściowych.4  
---

## **2\. Kompleksowa Analiza Architektury Danych (Exploratory Data Analysis)**

Zanim przystąpimy do zaawansowanej inżynierii, konieczne jest głębokie zrozumienie surowca, na którym operujemy. Zbiór danych Loan Prediction charakteryzuje się specyficzną mieszanką zmiennych, z których każda niesie unikalne wyzwania analityczne.

### **2.1. Zmienne Kategoryczne i Demograficzne: Ukryte Zależności**

Zmienne kategoryczne w tym zbiorze to nie tylko etykiety; to nośniki informacji socjologicznej i ekonomicznej.

* **Gender (Płeć):** Zmienna binarna (Male/Female). W kontekście modelowania ryzyka, jej użycie jest kontrowersyjne ze względów etycznych (fairness), jednak w historycznych danych często wykazuje korelacje z poziomem dochodów ze względu na lukę płacową. Analiza braków danych w tej kolumnie jest kluczowa – ich usunięcie może wprowadzić biks selekcji.  
* **Married (Stan Cywilny):** Zmienna ta (Yes/No) jest silnym predyktorem stabilności finansowej. Badania wskazują, że osoby w związkach małżeńskich często wykazują niższą skłonność do ryzyka i wyższą stabilność spłat, często dzięki istnieniu "poduszki finansowej" w postaci dochodów współmałżonka, nawet jeśli nie jest on oficjalnym współwnioskodawcą.5  
* **Dependents (Liczba Osób na Utrzymaniu):** Zmienna porządkowa (0, 1, 2, 3+). Wartość "3+" stanowi wyzwanie, gdyż zamienia zmienną numeryczną w kategoryczną. Z ekonomicznego punktu widzenia, każdy dodatkowy członek rodziny obniża dochód rozporządzalny (disposable income) na osobę, co teoretycznie zwiększa ryzyko defaultu przy stałym poziomie dochodów.2  
* **Education (Wykształcenie):** Podział na "Graduate" i "Not Graduate". Jest to proxy dla potencjału zarobkowego w przyszłości. Osoby z wyższym wykształceniem statystycznie szybciej awansują i są bardziej odporne na szoki na rynku pracy (np. bezrobocie strukturalne).  
* **Self\_Employed (Samozatrudnienie):** Kluczowy wskaźnik wariancji dochodów. Osoby samozatrudnione często mają wyższe, ale mniej stabilne dochody niż pracownicy etatowi. Modele ryzyka często traktują tę grupę bardziej rygorystycznie.

### **2.2. Zmienne Ciągłe i Ich Rozkłady: Problem Skośności**

Zmienne finansowe w zbiorze wykazują klasyczne cechy danych ekonomicznych – silną prawostronną skośność (right-skewness).

| Zmienna | Charakterystyka Rozkładu | Implikacje dla Modelowania |
| :---- | :---- | :---- |
| **ApplicantIncome** | Log-normalny, silne ogony (outliers) | Modele liniowe będą nadmiernie reagować na wartości ekstremalne. Wymagana normalizacja. |
| **CoapplicantIncome** | Duża liczba zer (Zero-inflated) | Wiele wniosków nie ma współwnioskodawcy. Należy rozważyć stworzenie flagi binarnej "Has\_Coapplicant". |
| **LoanAmount** | Prawostronnie skośny | Wartości puste w tej kolumnie są krytyczne – nie można udzielić kredytu bez kwoty. Imputacja musi uwzględniać dochód. |
| **Loan\_Amount\_Term** | Dyskretny, cykliczny (np. 360, 180\) | Traktowanie tego jako zmiennej ciągłej jest błędem. Są to standardowe okresy kredytowania (30 lat, 15 lat). Lepiej traktować jako zmienną kategoryczną lub porządkową. |

### **2.3. Credit\_History: Dominujący Sygnał**

Zmienna Credit\_History (0 lub 1\) jest bezsprzecznie najsilniejszym predyktorem w zbiorze.2 Reprezentuje ona skondensowaną historię przeszłych zachowań płatniczych. Korelacja tej zmiennej ze zmienną celu Loan\_Status często przekracza poziom 0.5. Jednakże, kluczowym problemem jest wysoki odsetek braków danych w tej kolumnie. W bankowości brak historii kredytowej (tzw. "thin file") nie jest tożsamy z brakiem danych (MCAR \- Missing Completely at Random). Jest to informacja sama w sobie – oznacza klienta, który nigdy wcześniej nie brał kredytu. Standardowa imputacja (np. modą) jest tutaj błędem metodologicznym, który zamazuje ten specyficzny profil ryzyka. Strategie Kaggle sugerują traktowanie braków jako oddzielnej kategorii lub predykcję tej wartości na podstawie innych cech.6

### **2.4. Anomalie i Braki Danych (Missing Values Analysis)**

Analiza wygrywających rozwiązań z hackathonów Analytics Vidhya wskazuje na różnorodne strategie radzenia sobie z brakami 1:

1. **Usuwanie:** W przypadku Loan\_ID czy wierszy z dużą liczbą braków, usuwanie jest akceptowalne, ale ryzykowne przy małym zbiorze danych.  
2. **Imputacja Statystyczna:** Wypełnianie średnią/medianą dla zmiennych ciągłych i modą dla kategorycznych. Jest to podejście bazowe (baseline).  
3. **Imputacja Modelowa:** Użycie IterativeImputer lub algorytmów typu k-NN do przewidywania brakujących wartości LoanAmount na podstawie ApplicantIncome i Education daje znacznie bardziej realistyczne wyniki, zachowując korelacje między zmiennymi.1

---

## **3\. Strategiczne Pozyskiwanie Danych Zewnętrznych (Data Enrichment)**

Aby przełamać "szklany sufit" wydajności modelu opartego wyłącznie na dostarczonych kolumnach, konieczne jest wzbogacenie zbioru o dane zewnętrzne. Ryzyko kredytowe nie istnieje w próżni; jest silnie skorelowane z otoczeniem makroekonomicznym i lokalnym.

### **3.1. Wskaźniki Makroekonomiczne: Kontekst Rynkowy**

Zdolność kredytowa jest funkcją nie tylko dochodu, ale i siły nabywczej pieniądza oraz kosztu kapitału. Chociaż zbiór danych jest anonimizowany czasowo, w rzeczywistych wdrożeniach (i symulacjach) kluczowe jest dołączenie następujących wskaźników 4:

* **Stopy Procentowe (Interest Rates):** Poziom stóp procentowych banku centralnego (np. WIBOR/LIBOR) w momencie udzielania kredytu determinuje wysokość raty (dla kredytów o zmiennej stopie). Wysokie stopy zwiększają ryzyko defaultu. Dodanie zmiennej Prevailing\_Interest\_Rate pozwala modelowi zrozumieć "ciężar" długu.  
* **Inflacja (CPI) i Dochód Realny:** Nominalny ApplicantIncome może być mylący. W środowisku wysokiej inflacji, dochód rozporządzalny (po opłaceniu kosztów życia) spada. Skorygowanie dochodu o wskaźnik inflacji (Real Income) pozwala na porównywanie wniosków z różnych okresów czasowych.8  
* **Stopa Bezrobocia (Unemployment Rate):** Regionalna stopa bezrobocia jest silnym stresorem. Klient z obszaru o wysokim bezrobociu strukturalnym jest bardziej narażony na utratę płynności finansowej. Można to powiązać ze zmienną Property\_Area.7  
* **PKB (GDP Growth):** Wzrost gospodarczy koreluje dodatnio ze spłacalnością kredytów. W okresach recesji (spadek PKB), korelacje między aktywami rosną (zjawisko zarażania się rynków), co drastycznie zwiększa PD portfela.

### **3.2. Dane Geoprzestrzenne i Wycena Nieruchomości**

Zmienna Property\_Area (Urban, Semiurban, Rural) jest zgrubnym przybliżeniem lokalizacji. Można ją jednak znacznie wzbogacić poprzez "probabilistyczne łączenie" (probabilistic merging) z zewnętrznymi danymi censusowymi.4

* **Płynność Aktywów:** Nieruchomości miejskie (Urban) zazwyczaj charakteryzują się wyższą płynnością i stabilniejszym wzrostem wartości niż wiejskie (Rural). Banki chętniej finansują aktywa, które łatwo zbyć w przypadku windykacji. Warto stworzyć cechę Asset\_Liquidity\_Score przypisującą wagi numeryczne tym kategoriom (np. Urban=1.2, Semiurban=1.0, Rural=0.8).  
* **Indeksy Cen Nieruchomości:** Jeśli możliwe jest oszacowanie regionu, nałożenie trendów cenowych (Housing Price Index) pozwala oszacować bieżące LTV (Loan-to-Value) w cyklu życia kredytu.  
* **Ryzyko Środowiskowe:** Wzbogacenie o dane ubezpieczeniowe dotyczące ryzyka powodziowego czy pożarowego dla danych typów lokalizacji. Obszary wiejskie mogą być bardziej narażone na klęski żywiołowe, co wpływa na wycenę zabezpieczenia.

### **3.3. Syntetyczna Generacja Danych (Synthetic Data Generation)**

W obliczu ograniczonej liczebności zbioru treningowego (co jest typowe dla udostępnionych próbek Analytics Vidhya), techniki generatywne stają się niezbędne.

* **SMOTE (Synthetic Minority Over-sampling Technique):** Zbiory kredytowe są z natury niezbalansowane (więcej spłacających niż niespłacających). SMOTE generuje syntetyczne przykłady klasy mniejszościowej poprzez interpolację liniową między istniejącymi sąsiadami w przestrzeni cech. Pozwala to "nauczyć" model charakterystyki defaultu bez prostego powielania wierszy (oversampling).9  
* **CTGAN (Conditional Tabular GANs):** Bardziej zaawansowaną metodą jest użycie sieci GAN (Generative Adversarial Networks) dedykowanych danym tabelarycznym. CTGAN uczy się rozkładu prawdopodobieństwa każdej kolumny (nawet tych o skomplikowanych, wielomodalnych rozkładach) i generuje zupełnie nowe, realistyczne rekordy klientów. Jest to szczególnie przydatne do trenowania głębokich sieci neuronowych (NN\_TORCH w AutoGluon), które wymagają dużych wolumenów danych.12  
* **ADASYN (Adaptive Synthetic Sampling):** Wariant SMOTE, który generuje więcej syntetycznych danych w pobliżu granicy decyzyjnej (tam, gdzie klasyfikacja jest najtrudniejsza), a mniej w głębi klastra klasy mniejszościowej. Pomaga to wyostrzyć granicę decyzyjną modelu.13

---

## **4\. Zaawansowana Inżynieria Cech (Feature Engineering)**

Sama surowa dana to za mało. "Inżynieria cech to sztuka wstrzykiwania wiedzy domenowej do modelu". W przypadku predykcji pożyczkowej, kluczowe są interakcje i wskaźniki finansowe.

### **4.1. Transformacje Matematyczne Zmiennych Ciągłych**

Jak wspomniano w sekcji 2.2, zmienne dochodowe są skośne. Modele oparte na odległościach (KNN, SVM) oraz liniowe (Logistic Regression) wymagają normalizacji rozkładu.

* **Transformacja Logarytmiczna:** Zastosowanie funkcji log(x+1) do ApplicantIncome i LoanAmount "ściska" ogon rozkładu, redukując wpływ wartości odstających. Jest to standardowy krok w wygrywających rozwiązaniach Kaggle.1 Przekształca to rozkład log-normalny w zbliżony do normalnego (Gaussa), co stabilizuje wariancję błędów (homoskedastyczność).  
* **Transformacja Boxa-Coxa i Yeo-Johnsona:** Są to bardziej elastyczne transformacje potęgowe, które automatycznie znajdują optymalny parametr lambda ($\\lambda$), aby maksymalnie zbliżyć rozkład do normalnego. Yeo-Johnson jest preferowany nad Box-Cox, ponieważ obsługuje wartości zerowe (częste w CoapplicantIncome) i ujemne, podczas gdy Box-Cox wymaga danych ściśle dodatnich.1

### **4.2. Konstrukcja Wskaźników Domenowych (Ratio Features)**

Najsilniejsze sygnały predykcyjne w bankowości to relacje (ratios) między dochodem a długiem. Model musi "widzieć" te relacje jawnie.

* Total Income (Dochód Gospodarstwa Domowego):

  $$\\text{Total\\\_Income} \= \\text{ApplicantIncome} \+ \\text{CoapplicantIncome}$$

  Suma dochodów jest często lepszym predyktorem niż dochód pojedynczego wnioskodawcy. Bezrobotna żona bogatego męża (jako współwnioskodawcy) ma wysoką zdolność kredytową, czego model nie zauważy patrząc tylko na jej ApplicantIncome \= 0.4  
* Loan-to-Income Ratio (LTI \- Wskaźnik Pożyczki do Dochodu):

  $$\\text{LTI} \= \\frac{\\text{LoanAmount}}{\\text{Total\\\_Income}}$$

  Wysokie LTI oznacza, że klient zadłuża się ponad miarę. Jest to fundamentalny wskaźnik ryzyka. W praktyce bankowej przekroczenie pewnego progu LTI dyskwalifikuje wniosek automatycznie.15  
* EMI (Equated Monthly Installment \- Szacowana Rata):

  $$\\text{EMI} \= \\frac{\\text{LoanAmount}}{\\text{Loan\\\_Amount\\\_Term}}$$

  Choć jest to uproszczenie (brak odsetek), daje to obraz miesięcznego obciążenia budżetu.  
* Debt-to-Income Ratio (DTI):

  $$\\text{DTI} \= \\frac{\\text{EMI}}{\\text{Total\\\_Income}}$$

  To jest "święty graal" oceny ryzyka. Jeśli DTI przekracza 40-50%, ryzyko defaultu rośnie wykładniczo. Modele drzewiaste mogą mieć trudność z samodzielnym "odkryciem" tej dzielonej zależności, dlatego należy ją podać wprost.4  
* Disposable Income (Dochód Rozporządzalny / Balance):

  $$\\text{Balance} \= \\text{Total\\\_Income} \- \\text{EMI}$$

  Kwota, która zostaje klientowi na życie. Nawet przy niskim DTI, jeśli Balance jest bliski zera (np. przy bardzo niskich dochodach), ryzyko jest wysokie.

### **4.3. Interakcje Zmiennych i Cechy Wielomianowe**

Czasami ryzyko wynika z kombinacji dwóch czynników, które osobno są niegroźne.

* **Credit History $\\times$ Property Area:** Klient ze złą historią kredytową na obszarze wiejskim (Rural) to znacznie wyższe ryzyko niż ten sam klient w mieście, gdzie łatwiej o pracę lub sprzedaż nieruchomości. Stworzenie cechy krzyżowej (np. Rural\_BadCredit, Urban\_GoodCredit) pomaga modelom szybciej wyizolować te segmenty.  
* Income Per Capita (Dochód na Głowę):

  $$\\text{Income\\\_Per\\\_Capita} \= \\frac{\\text{Total\\\_Income}}{\\text{Dependents} \+ 1}$$

  Wysoki dochód traci na znaczeniu, jeśli musi utrzymać dużą rodzinę. Normalizacja dochodu przez wielkość rodziny daje realny obraz zamożności.15  
* **Cechy Wielomianowe (Polynomial Features):** Generowanie kwadratów czy sześcianów zmiennych (np. ApplicantIncome^2) pozwala modelom liniowym uchwycić nieliniowe zależności (np. ryzyko maleje z dochodem, ale tylko do pewnego poziomu, powyżej którego stabilizuje się).14

### **4.4. Zaawansowane Kodowanie Zmiennych Kategorycznych**

Standardowe One-Hot Encoding (OHE) zwiększa wymiarowość i rzadkość danych (sparsity), co jest problematyczne dla niektórych algorytmów.

* **Target Encoding (Mean Encoding):** Zastąpienie kategorii (np. "Semiurban") średnią wartością zmiennej celu dla tej kategorii (np. 0.78 \- co oznacza 78% szans na akceptację). Jest to niezwykle silna technika, ale podatna na wyciek danych (data leakage). Musi być stosowana z regularyzacją (smoothing) i wewnątrz pętli walidacji krzyżowej.14  
* **Frequency Encoding:** Zastąpienie kategorii liczebnością jej występowania w zbiorze. Pozwala to modelowi zrozumieć, czy dana kategoria jest "typowa" czy "rzadka". Rzadkie kategorie (np. nietypowe wykształcenie, jeśli byłoby dostępne) mogą wiązać się z wyższym ryzykiem.1  
* Weight of Evidence (WoE): Technika wywodząca się ze statystyki aktuarialnej. Mierzy siłę dyskryminacyjną kategorii:

  $$\\text{WoE} \= \\ln \\left( \\frac{\\% \\text{Good}}{\\% \\text{Bad}} \\right)$$

  Ujemne WoE oznacza kategorię silnie skorelowaną z odrzuceniem wniosku. Zaletą WoE jest to, że tworzy liniową zależność z logarytmem szans (log-odds), co jest idealne dla regresji logistycznej.

---

## **5\. Architektury Modelowania i Wnioski z Kaggle (Kaggle Insights)**

Analiza publicznych repozytoriów kodów (kernels) z konkursów Kaggle i Analytics Vidhya ujawnia powtarzalne wzorce architektoniczne, które odróżniają rozwiązania przeciętne od zwycięskich.

### **5.1. Ensembling: Siła Różnorodności**

Pojedyncze modele rzadko wygrywają konkursy. Złotym standardem jest **Ensembling**, czyli łączenie predykcji wielu modeli.

* **Stacking (Agregacja):** Budowa meta-modelu (Level 2), który uczy się, jak najlepiej łączyć predykcje modeli bazowych (Level 1). Typowa zwycięska architektura to:  
  * *Level 1:* XGBoost, LightGBM, CatBoost (modele gradientowe) oraz Random Forest i Extra Trees (modele baggingowe) oraz sieć neuronowa (dla dywersyfikacji).  
  * *Level 2:* Regresja Logistyczna lub prosty model liniowy, który waży głosy modeli z poziomu 1\.  
  * Zaletą stackingu jest to, że modele bazowe popełniają błędy w różnych obszarach przestrzeni cech (np. drzewa są świetne na danych strukturalnych, sieci neuronowe mogą lepiej wychwycić nieliniowe interakcje po normalizacji), a meta-model uczy się to korygować.3  
* **Blending:** Prostsza forma, polegająca na uśrednianiu ważonym wyników (np. 0.4 \* XGB \+ 0.4 \* LGBM \+ 0.2 \* RF). Jest mniej podatna na overfitting niż stacking, ale często daje nieco słabsze wyniki.

### **5.2. Rola Random Forest w Świecie Gradient Boostingu**

Mimo dominacji XGBoost/LightGBM, Random Forest (RF) pozostaje kluczowym składnikiem ensembli. Dlaczego? Drzewa w RF są budowane niezależnie i głęboko (małe obciążenie, duża wariancja), a następnie uśredniane. Gradient Boosting buduje drzewa sekwencyjnie i płytko (duże obciążenie, mała wariancja). Ta fundamentalna różnica matematyczna sprawia, że błędy tych dwóch typów modeli są słabo skorelowane, co czyni je idealnymi kandydatami do łączenia.1

### **5.3. Obsługa Wartości Odstających (Outlier Treatment)**

W wielu zwycięskich notatnikach stosuje się agresywne przycinanie (clipping) lub wykluczanie górnego 1-5% wartości dla ApplicantIncome i LoanAmount. Wartości ekstremalne (tzw. wieloryby) mogą zdominować funkcję straty (loss function) modelu, zmuszając go do nadmiernego dopasowania się do tych kilku przypadków kosztem ogólnej generalizacji. Alternatywą jest użycie modeli odpornych na outliers, jak RobustScaler w preprocessingu.19

### **5.4. Pułapki Walidacyjne: Leakage**

Częstym błędem jest użycie Loan\_ID jako cechy. Choć wydaje się to oczywiste, w niektórych zbiorach ID są przydzielane sekwencyjnie, co sprawia, że ID niesie informację czasową (time leakage). Analiza wskazuje, że w tym zbiorze Loan\_ID jest losowe i powinno zostać bezwzględnie usunięte, aby uniknąć nauczenia się szumu. Ponadto, przy Target Encoding, kluczowe jest stosowanie walidacji krzyżowej (Stratified K-Fold), aby średnie były liczone na zbiorze treningowym (in-fold), a aplikowane do walidacyjnego (out-of-fold), unikając wycieku etykiety.20  
---

## **6\. Głęboka Optymalizacja AutoGluon: Poza zeroshot**

AutoGluon to potężne narzędzie AutoML, które automatyzuje proces trenowania, strojenia i łączenia modeli. Jednakże, domyślne ustawienia (presets='zeroshot' lub medium\_quality) są zoptymalizowane pod kątem szybkości i uniwersalności, a nie maksymalizacji wyniku na specyficznym, trudnym zbiorze danych kredytowych. Aby osiągnąć poziom ekspercki, musimy ingerować w proces automatyzacji.

### **6.1. Krytyka trybu hyperparameters='zeroshot'**

Preset zeroshot korzysta z portfolio pre-trenowanych konfiguracji, które historycznie sprawdzały się na setkach zbiorów danych. Jest to podejście statyczne – model nie przeszukuje aktywnie przestrzeni hiperparametrów dla *naszego* konkretnego zbioru danych, lecz wybiera "najlepsze z pudełka". W przypadku Loan Prediction, gdzie niuanse (jak wpływ Credit\_History czy skośność dochodu) są kluczowe, statyczne konfiguracje mogą być suboptymalne.21

### **6.2. Zaawansowany Bagging i Stacking (best\_quality)**

Pierwszym krokiem do poprawy wyników jest wymuszenie trybu presets='best\_quality'.

* **Bagging (Bootstrap Aggregating):** Zamiast pojedynczego podziału train/val, AutoGluon trenuje $k$ modeli na różnych fałdach (folds) danych (k-fold cross-validation) i uśrednia ich predykcje. Dla małych zbiorów danych (jak ten), bagging drastycznie redukuje wariancję predykcji i stabilizuje wynik. Zalecane jest użycie num\_bag\_folds=10, co pozwala wykorzystać 90% danych do treningu w każdej iteracji.  
* **Multi-layer Stacking:** AutoGluon potrafi budować wielopoziomowe stosy modeli. Włączenie num\_stack\_levels=2 lub 3 pozwala systemowi na trenowanie modeli, które jako wejście przyjmują wyjścia modeli z poprzedniej warstwy. To pozwala na uchwycenie niezwykle złożonych wzorców.22  
* **Rekomendacja Konfiguracyjna:**  
  Python  
  predictor.fit(...,   
                presets='best\_quality',   
                num\_stack\_levels=2,   
                num\_bag\_folds=10,  
                time\_limit=3600) \# Dłuższy czas jest niezbędny dla stackingu

### **6.3. Manualna Optymalizacja Hiperparametrów (HPO)**

Aby wyjść poza predefiniowane konfiguracje, należy zdefiniować własną przestrzeń przeszukiwania za pomocą hyperparameter\_tune\_kwargs. Wymusza to na AutoGluon aktywne przeszukiwanie (np. Optymalizacja Bayesowska lub Random Search).  
Strategia dla Klasyfikacji Binarnej (Loan Prediction):  
Należy skupić się na modelach Gradient Boosting (GBM, CatBoost) oraz sieciach neuronowych (NN\_TORCH), które korzystają z generowanych danych syntetycznych.

Python

\# Definicja strategii przeszukiwania  
hyperparameter\_tune\_kwargs \= {  
    'num\_trials': 50,          \# Duża liczba prób  
    'scheduler': 'local',      \# Lokalny planista  
    'searcher': 'auto',        \# Automatyczny wybór (często Bayesowski)  
}

\# Definicja przestrzeni hiperparametrów  
hyperparameters \= {  
    'GBM': {  
        'num\_boost\_round': 1000,  \# Duża liczba drzew  
        'learning\_rate': Float(0.01, 0.1, default=0.05), \# Wolne uczenie dla lepszej generalizacji  
        'feature\_fraction': Float(0.6, 1.0), \# Losowanie cech (przeciwdziałanie korelacji drzew)  
        'min\_data\_in\_leaf': Int(20, 50\) \# Zapobieganie overfittingowi na małym zbiorze  
    },  
    'NN\_TORCH': {  
        'num\_epochs': 50,  
        'learning\_rate': Real(1e-4, 1e-2, log=True),  
        'activation': Categorical(\['relu', 'gelu'\]),  
        'dropout\_prob': Real(0.0, 0.5) \# Dropout jest kluczowy dla regularyzacji  
    }  
}

predictor.fit(..., hyperparameters=hyperparameters, hyperparameter\_tune\_kwargs=hyperparameter\_tune\_kwargs)

Taka konfiguracja zmusza algorytmy gradientowe do budowania wielu drzew (1000) przy niskim współczynniku uczenia (learning\_rate), co jest znaną strategią wygrywającą ("slow learning is better learning").21

### **6.4. Kalibracja Progu Decyzyjnego (Decision Threshold Calibration)**

W problemach klasyfikacji binarnej domyślny próg odcięcia wynosi 0.5 (p \> 0.5 \-\> "Loan Approved"). Jednakże, koszt błędu jest asymetryczny. Udzielenie "złego" kredytu jest kosztowniejsze niż odrzucenie "dobrego".

* **Optymalizacja Post-Hoc:** AutoGluon oferuje metodę calibrate\_decision\_threshold(), która po wytrenowaniu modelu pozwala znaleźć optymalny próg dla zadanej metryki (np. F1-Score, który balansuje precyzję i czułość, lub Balanced Accuracy).  
* **Wniosek Praktyczny:** Często przesunięcie progu z 0.5 na np. 0.65 drastycznie zwiększa precyzję (Precision) kosztem niewielkiego spadku czułości (Recall), co w bankowości jest pożądanym kompromisem (konserwatywna polityka kredytowa).22

### **6.5. Wybór Metryki Ewaluacji**

Domyślna metryka accuracy (dokładność) jest myląca przy niezbalansowanych danych (np. 70% wniosków jest akceptowanych). Model, który akceptuje wszystkich ("All Yes"), miałby 70% dokładności, będąc bezużytecznym.

* **Rekomendacja:** Należy używać eval\_metric='roc\_auc' (obszar pod krzywą ROC) lub eval\_metric='f1\_macro'. ROC-AUC ocenia zdolność modelu do *rankingu* klientów od najlepszego do najgorszego, niezależnie od progu odcięcia, co jest standardem w ocenie ryzyka kredytowego.23

### **6.6. Niestandardowe Generatory Cech (Custom Feature Generators)**

Zamiast przetwarzać dane ręcznie w Pandas przed podaniem ich do AutoGluon, profesjonalne podejście polega na implementacji klasy dziedziczącej po AbstractFeatureGenerator. Pozwala to zaszyć logikę inżynierii cech (np. liczenie DTI) bezpośrednio w potoku (pipeline) AutoGluon. Gwarantuje to, że podczas predykcji na nowych danych (inference) te same transformacje zostaną zaaplikowane automatycznie, eliminując ryzyko błędów wdrożeniowych.21  
---

## **7\. Wnioski i Mapa Drogowa Wdrożenia**

Przeprowadzona analiza prowadzi do jednoznacznych konkluzji. Sukces w predykcji ryzyka na zbiorze Loan Prediction nie leży w ślepym zastosowaniu najpotężniejszego modelu, lecz w holistycznym podejściu do danych.  
**Syntetyczne Podsumowanie Rekomendacji:**

1. **Prymat Inżynierii nad Algorytmem:** Dobrze skonstruowane wskaźniki finansowe (**LTI, DTI, EMI, Total Income**) podane prostemu modelowi liniowemu często przewyższą surowy model XGBoost pozbawiony tej wiedzy domenowej. Relacje finansowe muszą być jawne.  
2. **Kontekst to Król:** Bez uwzględnienia otoczenia makroekonomicznego (nawet poprzez syntetyczne proxy inflacji czy stóp procentowych) model jest "krótkowzroczny".  
3. **AutoML jako Narzędzie Eksperckie, nie "Czarna Skrzynka":** Użycie AutoGluon w trybie presets='zeroshot' jest niewystarczające. Kluczem do wyników SOTA jest włączenie **Stackingu** (best\_quality), **Baggingu** oraz **Manualnej Kalibracji Hiperparametrów** (HPO) dla modeli gradientowych.  
4. **Zarządzanie Ryzykiem poprzez Progi:** Ostateczna decyzja kredytowa powinna opierać się na skalibrowanym progu prawdopodobieństwa, zoptymalizowanym pod kątem funkcji kosztu biznesowego (minimalizacja strat z defaultów), a nie domyślnej wartości 0.5.

Implementacja powyższych strategii przekształca standardowy problem akademicki w robustny system scoringowy, gotowy do symulacji w warunkach zbliżonych do produkcyjnych systemów bankowych.

#### **Works cited**

1. Loan Approval Prediction Machine Learning \- Analytics Vidhya, accessed on November 20, 2025, [https://www.analyticsvidhya.com/blog/2022/02/loan-approval-prediction-machine-learning/](https://www.analyticsvidhya.com/blog/2022/02/loan-approval-prediction-machine-learning/)  
2. Analytics Vidhya: Loan Prediction \- Kaggle, accessed on November 20, 2025, [https://www.kaggle.com/datasets/anmolkumar/analytics-vidhya-loan-prediction](https://www.kaggle.com/datasets/anmolkumar/analytics-vidhya-loan-prediction)  
3. Loan Prediction Problem From Scratch to End \- Analytics Vidhya, accessed on November 20, 2025, [https://www.analyticsvidhya.com/blog/2022/05/loan-prediction-problem-from-scratch-to-end/](https://www.analyticsvidhya.com/blog/2022/05/loan-prediction-problem-from-scratch-to-end/)  
4. Data-Driven Loan Default Prediction: A Machine Learning Approach for Enhancing Business Process Management \- MDPI, accessed on November 20, 2025, [https://www.mdpi.com/2079-8954/13/7/581](https://www.mdpi.com/2079-8954/13/7/581)  
5. rishabdhar12/Loan-Prediction-Analytics-Vidhya \- GitHub, accessed on November 20, 2025, [https://github.com/rishabdhar12/Loan-Prediction-Analytics-Vidhya-](https://github.com/rishabdhar12/Loan-Prediction-Analytics-Vidhya-)  
6. Feature Engineering in Machine Learning \- Analytics Vidhya, accessed on November 20, 2025, [https://www.analyticsvidhya.com/blog/2021/10/a-beginners-guide-to-feature-engineering-everything-you-need-to-know/](https://www.analyticsvidhya.com/blog/2021/10/a-beginners-guide-to-feature-engineering-everything-you-need-to-know/)  
7. using macroeconomic indicators to predict loan outcome \- Business, Management and Economics Engineering, accessed on November 20, 2025, [https://businessmanagementeconomic.org/pdf/Aman%20Upadhyay.pdf](https://businessmanagementeconomic.org/pdf/Aman%20Upadhyay.pdf)  
8. Advancing financial resilience: A systematic review of default prediction models and future directions in credit risk management \- PubMed Central, accessed on November 20, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11564005/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564005/)  
9. Smote for Imbalanced Classification with Python, Technique \- Analytics Vidhya, accessed on November 20, 2025, [https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/](https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/)  
10. SMOTE: Synthetic Data Generation for Balanced Datasets \- Lyzr AI, accessed on November 20, 2025, [https://www.lyzr.ai/glossaries/smote/](https://www.lyzr.ai/glossaries/smote/)  
11. Mastering Loan Default Prediction: Tackling Imbalanced Datasets for Effective Risk Assessment | by Utkarsh Lal | Geek Culture | Medium, accessed on November 20, 2025, [https://medium.com/geekculture/mastering-loan-default-prediction-tackling-imbalanced-datasets-for-effective-risk-assessment-8e8dfb2084d0](https://medium.com/geekculture/mastering-loan-default-prediction-tackling-imbalanced-datasets-for-effective-risk-assessment-8e8dfb2084d0)  
12. Synthetic Data Applications in Finance \- arXiv, accessed on November 20, 2025, [https://arxiv.org/pdf/2401.00081](https://arxiv.org/pdf/2401.00081)  
13. Loan Default Prediction \- Kaggle, accessed on November 20, 2025, [https://www.kaggle.com/code/vmuzhichenko/loan-default-prediction](https://www.kaggle.com/code/vmuzhichenko/loan-default-prediction)  
14. Advanced Feature Engineering for Machine Learning | by Silva.f.francis \- Medium, accessed on November 20, 2025, [https://medium.com/@silva.f.francis/advanced-feature-engineering-for-machine-learning-9e2e34c39a82](https://medium.com/@silva.f.francis/advanced-feature-engineering-for-machine-learning-9e2e34c39a82)  
15. FelixCharotte/LoanApprovalPrediction\_KaggleCompetition: Loan approval prediction Kaggle project \- GitHub, accessed on November 20, 2025, [https://github.com/FelixCharotte/LoanApprovalPrediction\_KaggleCompetition](https://github.com/FelixCharotte/LoanApprovalPrediction_KaggleCompetition)  
16. Data-Centric Machine Learning with Python \- School of Computing e-Library, accessed on November 20, 2025, [https://soclibrary.futa.edu.ng/books/Data-Centric%20Machine%20Learning%20with%20Python%20(%20etc.)%20(Z-Library).pdf](https://soclibrary.futa.edu.ng/books/Data-Centric%20Machine%20Learning%20with%20Python%20\(%20etc.\)%20\(Z-Library\).pdf)  
17. Intern Documwnt | PDF | Artificial Intelligence | Intelligence (AI) & Semantics \- Scribd, accessed on November 20, 2025, [https://www.scribd.com/document/882868992/Intern-Documwnt](https://www.scribd.com/document/882868992/Intern-Documwnt)  
18. Decision Tree vs Random Forest | Which Is Right for You? \- Analytics Vidhya, accessed on November 20, 2025, [https://www.analyticsvidhya.com/blog/2020/05/decision-tree-vs-random-forest-algorithm/](https://www.analyticsvidhya.com/blog/2020/05/decision-tree-vs-random-forest-algorithm/)  
19. Advanced Feature Engineering \- Kaggle, accessed on November 20, 2025, [https://www.kaggle.com/code/seneralkan/advanced-feature-engineering](https://www.kaggle.com/code/seneralkan/advanced-feature-engineering)  
20. Winning Solutions and Approaches from the Machine Learning Hikeathon \- Analytics Vidhya, accessed on November 20, 2025, [https://www.analyticsvidhya.com/blog/2019/04/ml-hikeathon-winning-solution-approaches/](https://www.analyticsvidhya.com/blog/2019/04/ml-hikeathon-winning-solution-approaches/)  
21. TabularPredictor.fit \- AutoGluon 1.4.1 documentation, accessed on November 20, 2025, [https://auto.gluon.ai/dev/api/autogluon.tabular.TabularPredictor.fit.html](https://auto.gluon.ai/dev/api/autogluon.tabular.TabularPredictor.fit.html)  
22. AutoGluon Tabular \- In Depth \- AutoGluon 1.4.0 documentation, accessed on November 20, 2025, [https://auto.gluon.ai/stable/tutorials/tabular/tabular-indepth.html](https://auto.gluon.ai/stable/tutorials/tabular/tabular-indepth.html)  
23. AutoGluon Tabular \- In Depth, accessed on November 20, 2025, [https://auto.gluon.ai/1.1.0/tutorials/tabular/tabular-indepth.html](https://auto.gluon.ai/1.1.0/tutorials/tabular/tabular-indepth.html)  
24. autogluon.tabular.models \- AutoGluon 1.4.1 documentation, accessed on November 20, 2025, [https://auto.gluon.ai/dev/api/autogluon.tabular.models.html](https://auto.gluon.ai/dev/api/autogluon.tabular.models.html)  
25. AutoGluon Predictors — AutoGluon Documentation 0.4.1 documentation, accessed on November 20, 2025, [https://auto.gluon.ai/0.4.1/api/autogluon.predictor.html](https://auto.gluon.ai/0.4.1/api/autogluon.predictor.html)  
26. autogluon.task — AutoGluon Documentation 0.0.14 documentation, accessed on November 20, 2025, [https://auto.gluon.ai/0.0.14/api/autogluon.task.html](https://auto.gluon.ai/0.0.14/api/autogluon.task.html)
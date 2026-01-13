# Product Guidelines - Kaggle ML Arena

## Design Principles
- **Technical Minimalism:** Interfejs CLI oraz dokumentacja powinny być zwięzłe i techniczne. Komunikaty muszą skupiać się na faktach, statusach i metrykach, unikając zbędnej gadatliwości.
- **Data-Driven Transparency:** Prezentacja wyników musi być przejrzysta i oparta na danych. Używamy tabel i uporządkowanych struktur do porównywania eksperymentów.

## Visual Identity (CLI)
- **Rich Formatting:** Wykorzystujemy bibliotekę `rich` do generowania czytelnych tabel podsumowujących eksperymenty, metryki i rankingi.
- **Semantic Coloring:** Stosujemy spójną kolorystykę dla statusów (np. sukces, ostrzeżenie, błąd), aby umożliwić błyskawiczną ocenę stanu procesów.
- **Focused Output:** Minimalizujemy szum w konsoli, dostarczając tylko kluczowe informacje, chyba że użytkownik zażąda trybu verbose.

## Brand Messaging & Tone
- **Core Message: "Total Reproducibility":** Każdy aspekt frameworka promuje i wymusza możliwość odtworzenia wyników. Podkreślamy niezawodność, wersjonowanie i automatyczne śledzenie zmian.
- **Tone:** Profesjonalny, precyzyjne i bezstronny. Narzędzie jest "cichym partnerem", który dba o infrastrukturę, pozwalając badaczowi skupić się na nauce.

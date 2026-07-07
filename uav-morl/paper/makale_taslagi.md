# Makale Taslağı

**Başlık (TR):** Dinamik Engelden Kaçınma ve Enerji Verimliliği için Uçtan Uca Çok Amaçlı Pekiştirmeli Öğrenme Tabanlı İHA Rota Planlama Çerçevesi

**Title (EN):** An End-to-End Multi-Objective Reinforcement Learning Framework for UAV Path Planning with Dynamic Obstacle Avoidance and Energy Efficiency

**Hedef mecralar:** IEEE Access, Drones (MDPI), Journal of Intelligent & Robotic Systems, Applied Soft Computing; ulusal: Gazi Üniv. Müh. Mim. Fak. Dergisi, Journal of Aviation.

---

## Özet (taslak)

İnsansız hava araçlarının (İHA) dinamik ortamlarda güvenli ve enerji-verimli rota planlaması, birbiriyle çelişen birden çok amacın eşzamanlı eniyilenmesini gerektirir. Bu çalışmada, hedefe ulaşma, enerji tüketimi ve çarpışma güvenliği amaçlarını vektörel bir ödül olarak modelleyen ve tek bir tercih-koşullu politika ile yaklaşık Pareto cephesini üretebilen uçtan uca bir çok amaçlı pekiştirmeli öğrenme (MORL) çerçevesi önerilmektedir. Döner kanatlı İHA'lar için analitik itki gücü modeli ödül fonksiyonuna gömülmüş; doğrusal, sinüzoidal ve dairesel hareket eden engeller içeren 3B bir benzetim ortamı geliştirilmiştir. PPO ve SAC algoritmalarıyla eğitilen politikalar, yeniden planlamalı A* ve yapay potansiyel alan (APF) taban çizgileriyle özdeş koşullarda karşılaştırılmıştır. Sonuçlar, önerilen çerçevenin ... başarı oranıyla ... daha az enerji tükettiğini ve tek politikadan test zamanında farklı enerji–güvenlik ödünleşim noktalarının elde edilebildiğini göstermektedir. *(Boşluklar deney sonuçlarıyla doldurulacak.)*

**Anahtar kelimeler:** İHA, rota planlama, çok amaçlı pekiştirmeli öğrenme, dinamik engelden kaçınma, enerji verimliliği, Pareto optimizasyonu, PPO, SAC

---

## 1. Giriş

- **Motivasyon:** Kargo, arama-kurtarma, tarım ve denetim uygulamalarında İHA'ların sınırlı batarya kapasitesi ve dinamik hava sahası (diğer araçlar, kuşlar, hareketli yapılar).
- **Boşluk (gap):** Literatürdeki RL tabanlı planlayıcıların çoğu (i) enerjiyi basit bir adım cezasıyla temsil eder (gerçek güç eğrisini kullanmaz), (ii) amaçları sabit ağırlıklarla tek skalara indirger (tercih değiştiğinde yeniden eğitim gerekir), (iii) statik engel varsayar.
- **Katkılar (madde madde):**
  1. Zeng & Zhang (2019) analitik döner kanat güç modelinin ödüle gömüldüğü fizik-bilgili (physics-informed) enerji amacı.
  2. Dirichlet örneklemeli tercih-koşullu MORL: tek politika → test zamanında ağırlık vektörüyle Pareto cephesi taraması.
  3. Üç hareket deseni içeren dinamik engel alanı ve ışın-döküm sensör modeli ile uçtan uca (algıdan ivme komutuna) öğrenme.
  4. PPO/SAC ile A*/APF'nin özdeş ortam ve metriklerle adil karşılaştırıldığı, tamamen tekrarlanabilir açık kaynak çerçeve.

## 2. İlgili Çalışmalar

- **2.1 Klasik rota planlama:** A*, D* Lite, RRT/RRT*, APF — dinamik ortam ve enerji zayıflıkları.
- **2.2 RL tabanlı İHA navigasyonu:** DQN/DDPG/PPO/SAC uygulamaları; tek amaçlı skalerleştirme eleştirisi.
- **2.3 Çok amaçlı RL:** Skalerleştirme aileleri (doğrusal, Chebyshev), Pareto-tabanlı yöntemler (Envelope Q-learning, PGMORL, CAPQL), tercih-koşullu politikalar.
- **2.4 İHA enerji modelleri:** Zeng & Zhang (2019) döner kanat; sabit kanat modelleri; enerji-farkındalıklı planlama çalışmaları.

## 3. Problem Formülasyonu

### 3.1 MOMDP tanımı
M = (S, A, P, **r**, γ), **r**(s,a) ∈ ℝ³.

### 3.2 İHA kinematiği
v_{t+1} = (1−λ) v_t + a_t Δt (hız sınırı v_max = 8 m/s, ivme sınırı a_max = 4 m/s², Δt = 0.2 s, sürtünme λ = 0.05). *(Kod: `src/envs/uav_env.py`, step)*

### 3.3 Enerji modeli
P(V) = P₀(1 + 3V²/U_tip²) + P_i(√(1 + V⁴/4v₀⁴) − V²/2v₀²)^{1/2} + ½d₀ρsAV³ + mgV_z⁺
*(Kod: `src/envs/energy.py`; parametre tablosunu makaleye Tablo 1 olarak koyun.)*

### 3.4 Gözlem, eylem, ödül
- Gözlem (27B): göreli hedef (3) + hız (3) + 16 ışın + enerji (1) + min. mesafe (1) + w (3).
- Ödül bileşenleri: r_hedef (ilerleme + varış bonusu 50), r_enerji (−0.1·E_adım/E_hover), r_güvenlik (tehlike bölgesi karesel cezası + çarpışma −50).
- Doğrusal skalerleştirme: R = w·r, w ~ Dirichlet(1,1,1) (eğitim), w sabit (test/Pareto).

### 3.5 Sonlanma koşulları
Hedef (<2 m), çarpışma, batarya (60 kJ), 400 adım.

## 4. Yöntem

- **4.1 Tercih-koşullu politika mimarisi:** π(a|s,w); w'nin gözleme eklenmesi; neden tek ağ ile Pareto taraması mümkün (Yang et al. 2019 envelope argümanına atıf).
- **4.2 Algoritmalar:** PPO (clip=0.2, 8 paralel ortam, 2×256 MLP) ve SAC (buffer 5×10⁵). Hiperparametre tablosu: `configs/default.yaml`'dan Tablo 2.
- **4.3 Eğitim protokolü:** 10⁶ adım × 3 tohum; her 25k adımda 20 bölümlük değerlendirme; en iyi model saklama.
- **4.4 Taban çizgileri:** (i) 2 m çözünürlüklü 3B ızgarada her 5 adımda yeniden planlayan A*; (ii) Khatib APF (k_att=1, k_rep=30, ρ₀=6). Her ikisi de aynı ivme arayüzünden ortama bağlanır → adil karşılaştırma.

## 5. Deneysel Kurulum

- Arena 60×60×20 m, 10 engel (yarıçap 1.5–3.5 m, hız 0.5–2.5 m/s, %70 dinamik), min. başlangıç-hedef 40 m.
- Değerlendirme: yöntem başına 100 bölüm, eğitimde görülmemiş tohumlar.
- Pareto taraması: 8 ağırlık vektörü (configs → evaluation.pareto_weights).
- Donanım/istatistik: ortalama ± std, 3 tohum; Welch t-testi veya Mann-Whitney U ile anlamlılık (p<0.05) raporlayın.

## 6. Sonuçlar (deneyler bittikten sonra doldurulacak)

- **Tablo 3:** Yöntem karşılaştırması (SR, CR, E, E/m, L, verimlilik, min. mesafe). → `results/metrics/*_summary.json`
- **Şekil 1:** Ortam şeması ve gözlem/eylem mimarisi (elle çizilecek blok diyagram).
- **Şekil 2:** Öğrenme eğrileri (3 tohum, gölgeli std). → `learning_curves.png`
- **Şekil 3:** Örnek 3B yörüngeler. → `trajectories_3d.png`
- **Şekil 4:** Pareto cephesi (enerji–rota, enerji–güvenlik). → `pareto_front.png`
- **Şekil 5:** Yöntem karşılaştırma çubukları. → `method_comparison.png`
- **6.x Ablasyonlar (önerilir, hakem sorar):**
  - Enerji modeli yerine basit hız cezası → enerji metriğine etkisi
  - Tercih koşullama kapalı (sabit w) → Pareto kapsamı
  - Işın sayısı 8/16/32 → başarı oranı
  - Dinamik engel oranı %0/%50/%100 → çarpışma oranı

## 7. Tartışma

- Tek politika ile ödünleşim kontrolünün pratik değeri (görev sırasında w değişimi).
- APF'nin dinamik ortamdaki reaktif avantajı vs. yerel minimum zaafı; A*'ın donmuş-dünya varsayımı.
- Sınırlamalar: kinematik (dinamik değil) model, mükemmel algı varsayımı, rüzgârsız ortam, sim-to-real boşluğu.

## 8. Sonuç ve Gelecek Çalışmalar

Çok-ajanlı genişletme, rüzgâr alanı, gerçek uçuş testi (PX4/Gazebo SITL), Chebyshev skalerleştirme ve hypervolume metriği ile MORL kıyaslaması.

---

## Yazım Süreci Yol Haritası

1. `scripts\train_all_seeds.bat` ile deneyleri koşun (GPU'da ~saatler, CPU'da ~1 gün).
2. Her tohum için `evaluate` + `--pareto` çalıştırın; CSV'leri birleştirip ortalama±std hesaplayın.
3. Ablasyonlar için `configs/default.yaml` kopyalayıp ilgili parametreyi değiştirin (ör. `configs/ablation_no_energy.yaml`).
4. Figürleri `results/figures`'tan alın (150 DPI; dergi isterse 300 DPI için `plots.py` içindeki `figure.dpi` değerini yükseltin).
5. `paper/main.tex` iskeletini doldurun; Overleaf'e taşıyabilirsiniz.

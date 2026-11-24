"""
Funktioner för analys av hälsodata.

Modulen innehåller funktioner för beskrivande statistik, simulering, 
konfidensintervall, hypotesprövning och powersimulering.
"""

import numpy as np
from scipy import stats

def descriptive_stats(df, cols):
    """
    Beräknar medelvärde, median, min och max för valda kolumner
    
    Parametrar
    ----------
    
    df : pandas.DataFrame
        Datan som ska analyseras.
    cols : list of str
        Kolumnerna som statistiken ska beräknas för.
    
    Returnerar
    ----------
    pandas.DataFrame
        En tabell med statistiken, avrundad till två decimaler.
    """
    return df[cols].agg(["mean", "median", "min", "max"]).round(2)

def simulate_disease_share(p, n, rng):
    """
    Simulerar n personer som var och en har sjukdomen med sannolikheten p.
    
    Parametrar
    ----------
    p : float
        Sannolikheten för sjukdom.
    n : int
        Antal simulerade personer.
    rng : numpy.random.Generator
        Slumpgenerator med seed, för reproducerbara resultat.
        
    Returnerar
    ----------
    p_sim : float
        Andelen sjuka i simuleringen.
    se : float
        Förväntat standardfelet för andelen, sqrt(p(1-p)/n).
    """
    simulated = rng.random(n) < p
    p_sim = float(simulated.mean())
    se= float(np.sqrt(p*(1-p)/n))
    return p_sim, se

def ci_mean_normal(x, confidence = 0.95):
    """
    konfidensintervall för medelvärdet med normalapproximation:
    medel ± z * (sd / sqrt(n)).
    
    Parametrar
    ----------
    x : array-like
        Datan.
    confidence : float, standard 0.95
        Konfidensnivå.
        
    Returnerar
    ----------
    lo, hi : float
        Nedre och övre gräns för intervallet.
    mean_x = float
        Medelvärdet.
    sd_x : float
        Standardavvikelsen(ddof = 1).
    n : int
        Antal observationer.
    """
    x = np.asarray(x, dtype=float)
    mean_x = float(np.mean(x))
    sd_x = float(np.std(x, ddof=1))
    n = len(x)
    
    z_critical = stats.norm.ppf((1 + confidence) / 2) #1,96 för 95%
    half_width = z_critical * sd_x / np.sqrt(n)  
    return mean_x - half_width, mean_x + half_width, mean_x, sd_x, n

def ci_mean_bootstrap(x, rng, B=10_000, confidence=0.95):
    """
    Konfidensintervall för medelvärdet med bootstrap(percentilmetoden).
    
    Parametrar
    ----------
    x : array-like
        Datan
    rng : numpy.random.Generator
        Slumpgenerator med seed.
    B : int, standard 10 000
        Antal bootstrap-stickprov.
    confidence : float,standard 0.95
        Konfidensnivå.

    Returnerar
    ----------
    lo, hi : float
        Nedre och övre gräns för intervallet.
    mean_x : float
        Medelvärdet av originaldatan
    boot_means : numpy.ndarray
        Alla B bootstrap-medelvärden.
    """
    x = np.asarray(x, dtype = float)
    n = len(x)
    boot_means = np.empty(B)
    for b in range(B):
        boot_sample = rng.choice(x, size = n, replace = True)
        boot_means[b] = np.mean(boot_sample)

    alpha = (1 - confidence) / 2
    lo, hi = np.percentile(boot_means, [100*alpha, 100*(1-alpha)])
    return float (lo), float(hi), float(np.mean(x)), boot_means 

def welch_ttest(group1, group2, alternative = "greater"):
    """
    Welchs t-test för två oberoende grupper(förutsätter inte lika varians).
    
    Parametrar
    ---------
    group1, group2 : array-like
        De två grupperna som jämförs.   
    alternative : str, standard "greater"
        "greater" testar om group1 har högre medelvärde än group2
        
    Returnerar
    ----------
    t_stat : float
        t-värdet.
    p_value : float
        p-värdet.
    """
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var = False,
                                      alternative = alternative)
    return float(t_stat), float(p_value)


def run_one_t_test(n1, n2, true_diff, sd, base_mean, rng):
    """En simulerad studie: skapar normalfördelade blodtryck för två grupper
    och kör Welchs t-test.

    Returnerar
    ---------
    float 
        p-värdet från testet.
    """
    group1_sim = rng.normal(loc=base_mean + true_diff, scale=sd, size=n1)
    group2_sim = rng.normal(loc=base_mean, scale=sd, size=n2)
    _, p = welch_ttest(group1_sim, group2_sim, alternative="greater")
    return p


def estimate_power_by_simulation(n1, n2, true_diff, sd, base_mean, rng, 
                                 alpha=0.05, num_simulations=2000):
    """
    Uppskattar power genom att simulera många studier och räkna 
    hur ofta testet hittar skillnaden(p < alpha).
    
    Returnerar
    ---------
    float
        Andelen simuleringar där skillnaden hittades (power).
    """
    detections = 0
    for _ in  range(num_simulations):
        p = run_one_t_test(n1, n2, true_diff, sd, base_mean, rng)
        if p < alpha:
            detections += 1
    return detections / num_simulations

def fit_linear_regression(df, x_cols, y_col):
    """Anpassar en linjär regression med minsta kvadratmetoden (numpy). 
    
    Modellen skrivs i matrisform som y = X . beta, där X har en kolumn med ettor (för interceptet) och en kolumn per förklarande variabel.
    
    Parametrar
    ----------
    df : pandas.DataFrame
        Datan
    x_cols : list of str
        De förklarande variablerna, t.ex. ["age].
    y_cols : str
        Variabeln som ska förutsägas, t.ex. "systolic_bp".

    Returnerar
    ----------
    coefs : dict
        Interceptet och en koefficient per förklarande variabel.
    r2 : float
        Förklaringsgraden R².
    y_pred : numpy.ndarray
        Modellens förutsagda värden.
    """
    X = np.column_stack([np.ones(len(df)), df[x_cols].to_numpy(dtype=float)])
    y = df[y_col].to_numpy(dtype=float)

    beta,*_ = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot 

    coefs = dict(zip(["intercept"] + list(x_cols), beta))
    return coefs, float(r2), y_pred

def two_proportion_ztest(x1, x2):
    """
    z-test för skillnaden mellan två andelar (tvåsidigt).
    
    Används för ja/nej-variabler (0/1), t.ex om andelen sjuka skiljer sig
    mellan två grupper.
    
    Parametrar
    ----------
    x1, x2 : array - like
        0/1-värden för grupp 1 och grupp 2.

    Returnerar
    ----------
    p1, p2 : float
        Andelen ettor i resppektive grupp.
    z : float
        z-värdet.
    p_value : float
        Tvåsidigt p_värde.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1, n2 = len(x1), len(x2)
    p1, p2 = x1.mean(), x2.mean()

    p_pool = (x1.sum() + x2.sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_value = 2 * stats.norm.sf(abs(z))

    return float(p1), float(p2), float(z), float(p_value)
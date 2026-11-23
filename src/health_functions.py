




"""
Funktioner fäör analys av hälsodata.

Modulen innehåller funktioner för beskrivande statistik, simulering, 
konfidensinterval, hypotesprövning och powersimulering.
"""

import numpy as np
from scipy import stats

def descriptive_stats(df, cols):
    """
    Beräknar medelvärde, median, min och max för valda kolumner
    
    Parametrar
    ----------
    
    df:pandas.DataFrame
        Datan som ska analyseras.
    cols: list of str
        Kolumnerna som statistiken ska beräknas för.
    
    Returnerar
    
    ----------
    pandas.DataFrame
        n tabell med statistiken, avrunda till två decimaler.
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
    rng :numpy.random.Generator
        Slumpgenerator med seed, för reprducerbara resultat.
        
    Returnerar
    ----------
    
    p_sim : float
        Andelen sjuka i simuleringen.
    se : float
        Förväntat standardfelbför andelen, sqrt(p(1-p)/n).
    """
    simulated = rng.random(n) < p
    p_sim = float(simulated.mean())
    se= float(np.sqrt(p*(1-p)/n))
    return p_sim, se

def ci_mean_normal(x, confidence = 0.95):
    """
    Konfidensinterval för medelvärdet med normalapproximation:
    medel ± z * (sd / sqrt(n)).
    
    Parametrar
    ----------
    x : array - like
        Datan.
    confidence : float, standard 0.95
        Konfidensnivå.
        
    Returnerar
    ----------
    lo, hi : float
        Medelvärdet.
    sd_x = float
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
    
    PAramatrar
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
    boot_means = numpy.ndarray
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
        De två grupperna jämförs.
    alternative : str, standard "greater"
        "greater" tester om group1 har högre medelvärde än group2
        
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
    for kör Welchs t-test
    
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
        Andelen simulerinagr där skillnaden hittades (power).
    """
    detections = 0
    for _ in  range(num_simulations):
        p = run_one_t_test(n1, n2, true_diff, sd, base_mean, rng)
        if p < alpha:
            detections += 1
    return detections / num_simulations

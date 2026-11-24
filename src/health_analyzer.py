"""
Klassen HealthAnalyzer för analys av hälsodata.
"""

import matplotlib.pyplot as plt

from .health_functions import descriptive_stats,welch_ttest

class HealthAnalyzer:
    """
    Samlar hälsodatan och metoder för att analysera och visulasera den.
    
    Parametrar
    ----------
    df : pandas.DataFrame
    Datasetet med hälsodata.
    
    Exempel
    -------
    >>> analyzer = HealthAnalyzer(df)
    >>> analyzer.summary(["age", "systolic_bp"])
    """

    def __init__(self, df):
        """Sparar datan i objektet så att alla metoder kan använda den."""
        self.df = df

    def __repr__(self):
        """Visar en kort beskrivning av objektet"""
        return f"HealthAnalyzer(n_deltagare={len(self.df)}, kolumner = {list(self.df.columns)})"

    @property
    def n_participants(self):
        """Antal deltagare i datasetet."""
        return len(self.df)

    def summary(self, cols):
        """Beräknar medelvärde, median, min och max för valda kolumner.
        
        Parametrar
        ----------
        cols : list of str
            Kolumnerna som statistiken ska beräknas för.
            
        Returnerar
        ----------
        pandas.DataFrame
            Tabell med statistiken.
        """
        return descriptive_stats(self.df, cols)

    def compare_groups(self, value_col, group_col, group1, group2,
                       alternative = "greater"):
        """
        Jämför medelvärdet av en variabel mellan två grupper med Welchs t-test.
        
        Parametrar
        ----------
        value_col : str
            Variabeln som jämförs, t.ex. "systolic_bp".
        group_col : str 
            Kolumnen som delar in grupperna, t.ex. "smoker".
        group1, group2 : str
            Värdena för de två grupperna, t.ex "Yes" och "No".
        alternative : str, standard "greater"
            "greater" testar om group1 har högre medelvärde än group2.
            "two-sided" testar om grupperna skiljer sig åt i någon riktning.
            
        Returnerar
        ----------
        dict
            Medelvärden, skillnad, t-värde och p-värde.
        """

        g1 = self.df.loc[self.df[group_col] == group1, value_col]
        g2 = self.df.loc[self.df[group_col] == group2, value_col]
        t_stat, p_value = welch_ttest(g1, g2, alternative=alternative)

        return{
            "medel_grupp1" : g1.mean(),
            "medel_grupp2" : g2.mean(),
            "skillnad" : g1.mean() - g2.mean(),
            "t" : t_stat,
            "p" : p_value,
        }

    def plot_histogram(self, col, bins = 20, ax = None):
        """
        Ritar ett histogram för en kolumn, med linjer för medelvärde och median.
        
        Parametrar
        ----------
        col : str
            Kolumnen som ska visas.
        bins : int, standard 20
            Antal staplar.
        ax : matplotlib.axes.Axes, valfri
            Axel att rita i. Om ingem anges skapas en ny figur.
            
        Returnerar
        ----------
        matplotlib.axes.Axes
            Axeln med grafen.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,5))

        data = self.df[col]
        ax.hist(data, bins=bins, edgecolor="black", alpha=0.7)
        ax.axvline(data.mean(), color = "tab:red", linestyle = "--",
                   label = f"Medelvärde ({data.mean():.1f})")
        ax.axvline(data.median(), color = "tab:green", linestyle = ":", linewidth = 2,
                   label = f"Median ({data.median():.1f})")
        ax.set_title(f"Fördelning av {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Antal deltagare")
        ax.grid(True, axis="y")
        ax.legend()
        return ax
    
"""
Klassen HealthAnalyzer för analys av hälsodata.
"""

import matplotlib.pyplot as plt
import numpy as np

from .health_functions import(
    descriptive_stats,
    welch_ttest, 
    two_proportion_ztest,
    fit_linear_regression,
    compare_regression_models
)

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
        self.results = {}

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

    def rate_by_group(self, group_cols, target_col = "disease"):
        """
        Beräknar antal personer och andelen med target_col = 1 per grupp.
        
        Paraametrar
        -----------
        group_cols : str eller list of str
            Kolumnen eller kolumnerna som grupperna bildas av.
        target_col : str, standard "disease"
            en 0/1-kolumn.   

        Returnerar
        ----------
        pandas.DataFrame
            Antal och andel per grupp.
        """
        return (self.df.groupby(group_cols)[target_col]
                .agg(antal="size", andel="mean")
                .round(3))

    def compare_rates(self, group_col, group1, group2, target_col="disease"):
        """
        Jämför andelen med target_col = 1 mellan två grupper med ett z-test.
        
        Retunerar
        ---------
        dict
            Andelar, skillnad, z-värde och p-värde.
        """
        x1 = self.df.loc[self.df[group_col] == group1, target_col]
        x2 = self.df.loc[self.df[group_col] == group2, target_col]
        p1, p2, z, p_value = two_proportion_ztest(x1, x2)

        return {
            "andel_grupp1": p1,
            "andel_grupp2": p2,
            "skillnad" : p1 - p2,
            "z" : z,
            "p" : p_value,
        }

    def plot_rate_by_group(self, group_col, hue_col, target_col = "disease", ax = None):
        """
        Staplediagram över andelen med target_col = 1, uppdelat på två kolumner.
        
        Parametrar
        ----------
        group_col : str
            Kolumnen spå x-axeln, t.ex. "sex".
        hue_col : str
            Kolumnen som ger staplar sida vid sida, t-ex. "smoker"
        target_col : str, standard "disease"
            En 0/1-kolumn.
        ax : matplotlib.axes.Axes, valfri
            Axel att rita i.
        
        
        Retunerar
        ---------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize = (8,5))

        table = self.df.pivot_table(index=group_col, columns=hue_col,
                                    values=target_col, aggfunc="mean") * 100
        table.plot (kind="bar", ax=ax, edgecolor="black",rot=0)

        ax.set_title(f"Andel med {target_col} per {group_col} och {hue_col}")
        ax.set_xlabel(group_col)
        ax.set_ylabel("Andel (%)")
        ax.grid(True, axis="y")
        ax.legend(title = hue_col)
        return ax

    def fit_regression(self, x_cols, y_col="systolic_bp"):
        """
        Anpassar en linjär regression och sparar resultatet i self.results.
        
        Parametrar
        ----------
        x_cols : list of str
            De förklarande variablerna, t.ex.["age","weight"]
        y_col : str, standard "systolic_bp"
            Variabeln som ska förutsägas.

        Returnerar
        ----------
        coefs : dict
            Interceptet och koefficienterna.
        r2 : float
            Förklaringsgraden R
        y_pred : numpy.ndarray
            Modellens förutsagda Värden.  
        """
        coefs, r2, y_pred = fit_linear_regression(self.df, x_cols, y_col)

        model_name = f"{y_col} ~ {' + '.join(x_cols)}"
        self.results[model_name] = {"koefficienter": coefs, "R2" : r2}

        return coefs,r2,y_pred

    def compare_models(self, models, y_col = "systolic_bp"):
        """
        Jämföra flera regressionsmodeller med R², justerat R² och RMSE.
        
        Parametrar
        ----------
        models : dict
            Modellnamn som nycklar och listor med variabler som värden.
        y_col : str, standard "systolic_bp"
            Variabeln som ska förutsägas.

        Returnerar
        ----------
        pandas.DataFrame
            En rad per modell.
        """
        return compare_regression_models(self.df, models, y_col)

    def plot_regression(self, x_col, y_col="systolic_bp", ax=None):
        """
        Spridningsdiagram med regressionslinje för en förklarande variabel.

        Parametrar
        ----------
        x_col : str
            Variabeln på x-axeln, t.ex. "age".
        y_col : str, standard "systolic_bp"
            Variabeln på y-axeln.
        ax : matplotlib.axes.Axes, valfri
            Axel att rita i.

        Returnerar
        ----------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        coefs, r2, _ = self.fit_regression([x_col], y_col)

        ax.scatter(self.df[x_col], self.df[y_col], alpha=0.4, label="Deltagare")
        x_line = np.linspace(self.df[x_col].min(), self.df[x_col].max(), 100)
        y_line = coefs["intercept"] + coefs[x_col] * x_line
        ax.plot(x_line, y_line, color="tab:red", linewidth=2,
                label=f"Regressionslinje (R² = {r2:.2f})")

        ax.set_title(f"{y_col} och {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True)
        ax.legend()
        return ax

    def plot_predicted_vs_actual(self, x_cols, y_col="systolic_bp", ax=None):
        """
        Jämför verkliga och förutsagda värden för en regressionsmodell.

        Om modellen vore perfekt skulle alla punkter ligga på den
        streckade linjen y = x.

        Parametrar
        ----------
        x_cols : list of str
            De förklarande variablerna.
        y_col : str, standard "systolic_bp"
            Variabeln som förutsägs.
        ax : matplotlib.axes.Axes, valfri
            Axel att rita i.

        Returnerar
        ----------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        _, r2, y_pred = self.fit_regression(x_cols, y_col)
        y_true = self.df[y_col]

        ax.scatter(y_true, y_pred, alpha=0.4)
        lims = [y_true.min(), y_true.max()]
        ax.plot(lims, lims, color="tab:red", linestyle="--", label="Perfekt modell (y = x)")

        ax.set_title(f"Verkligt mot förutsagt {y_col} (R² = {r2:.2f})")
        ax.set_xlabel(f"Verkligt {y_col}")
        ax.set_ylabel(f"Förutsagt {y_col}")
        ax.grid(True)
        ax.legend()
        return ax
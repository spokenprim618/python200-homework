import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import ttest_ind
import os

@task(retries=3, retry_delay_seconds=2)
def load_Data():
    main_df = pd.DataFrame(columns = ['Year'])
    for file in os.listdir("data"):
        full_path = os.path.join("data", file)
        print(full_path)
        sub_df = pd.read_csv(full_path,sep=";",decimal=",")
        year_used = int(file[16:20])
        if "Ladder score" in sub_df.columns:
            sub_df = sub_df.rename(columns={"Ladder score": "Happiness score"})

        sub_df['Year'] = year_used
        main_df = pd.concat([main_df, sub_df], ignore_index=True)
    
    completed_path = os.path.join("./outputs","WHOutput.csv")
    main_df.to_csv(completed_path,sep=";",decimal=",",index=False)

    print(f"Completed path: {completed_path}")

    total_countries = main_df["Country"].nunique()
    total_years = main_df["Year"].nunique()
    return {"total_countries": total_countries,"total_years": total_years}

@task
def des_stats(file):
    logger = get_run_logger()
    df = pd.read_csv(file, sep=";", decimal=",")

    infoS = df['Happiness score'].agg(["mean", "median", "std"])
    logger.info(f"Happiness score information\n{infoS}")

    infoy = df.groupby('Year')['Happiness score'].agg(["mean", "median", "std"])
    logger.info(f"Happiness score information by year\n{infoy}")

    infor = df.groupby('Regional indicator')['Happiness score'].agg(["mean", "median", "std"])
    logger.info(f"Happiness score information by region\n{infor}")

    output = (
    f"Happiness score information\n{infoS.to_string()}\n\n"
    f"Happiness score information by year\n{infoy.to_string()}\n\n"
    f"Happiness score information by region\n{infor.to_string()}\n"
    )
    
    with open("personal_output/descriptive_stats.txt", "w") as f:
        f.write(output)
    
    top_regions = infor.sort_values(by="mean",ascending=False).head(3)

    bottom_regions = infor.sort_values(by="mean",ascending=True).head(3)

    return {"top_3_regions": top_regions,"bottom_3_regions": bottom_regions}

@task
def visualizations(file):
    df = pd.read_csv(file,sep=";",decimal=",")

    #Histogram

    for year, group in df.groupby('Year'):
        plt.hist(group['Happiness score'], alpha=0.5, label=str(year))
        plt.legend()    
    plt.title("Dist of Happiness Scores")
    plt.xlabel("Happiness Score by Year")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join("outputs", "happiness_histogram.png"))
    plt.close()

    # Box Plot
    grouped = df.groupby('Year')['Happiness score']
    data = [group.dropna() for _, group in grouped]
    labels = [str(year) for year, _ in grouped]
    plt.boxplot(data,label=labels)
    plt.title("Happiness by Year")
    plt.savefig(os.path.join("outputs", "happiness_by_year.png"))
    plt.close()

    # Scatter plot
    plt.scatter(df['GDP per capita'],df['Happiness score'], color="green")
    plt.title("GDP vs happiness")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness score")
    plt.legend()
    plt.savefig(os.path.join("outputs", "gdp_vs_happiness.png"))
    plt.close()

    # Heatmap
    numeric_df = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr(method="pearson")
    sns.heatmap(corr_matrix, annot = True)
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join("outputs", "correlation_heatmap.png"))
    plt.close()


@task
def hypothesis_tests(file):

    df = pd.read_csv(file, sep=";", decimal=",")
    logger = get_run_logger()

    scores_2019 = df[df["Year"] == 2019]["Happiness score"].dropna()
    scores_2020 = df[df["Year"] == 2020]["Happiness score"].dropna()

    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()

    t_stat, p_value = ttest_ind(scores_2019,scores_2020)

    logger.info(f"2019 mean happiness: {mean_2019:.3f}")
    logger.info(f"2020 mean happiness: {mean_2020:.3f}")
    logger.info(f"t-statistic: {t_stat:.3f}")
    logger.info(f"p-value: {p_value:.6f}")

    if p_value < 0.05:

        if t_stat > 0:
            interpretation = (
                f"The average happiness score in 2020 was significantly lower "
                f"than in 2019 (t={t_stat:.3f}, p={p_value:.4f}). "
                "This may suggest that the beginning of the COVID-19 pandemic "
                "could be associated with a decline in global happiness between these 2 years."
            )
        else:
            interpretation = (
                f"The average happiness score in 2020 was significantly higher "
                f"than in 2019 (t={t_stat:.3f}, p={p_value:.4f}). "
                "This may suggest happiness levels changed "
                "between the two years despite the pandemic positvely."
            )

    else:
        interpretation = (
            "There is not enough statistical evidence to conclude that the average "
            "global happiness was different between 2019 and 2020. Any observed "
            "difference could reasonably be due to random variation because of "
            "how small the differnce is."
    )

    logger.info(interpretation)

    europe = df[df["Regional indicator"] == "Western Europe"]["Happiness score"].dropna()

    africa = df[ df["Regional indicator"] == "Sub-Saharan Africa"]["Happiness score"].dropna()

    europe_mean = europe.mean()
    africa_mean = africa.mean()

    t_stat2, p_value2 = ttest_ind(europe,africa)

    logger.info(
        f"Western Europe mean happiness: {europe_mean:.3f}"
    )

    logger.info(
        f"Sub-Saharan Africa mean happiness: {africa_mean:.3f}"
    )

    logger.info(
        f"Region comparison t-statistic: {t_stat2:.3f}"
    )

    logger.info(
        f"Region comparison p-value: {p_value2:.6f}"
    )
    output = []

    output.append(f"2019 mean happiness: {mean_2019:.3f}")
    output.append(f"2020 mean happiness: {mean_2020:.3f}")
    output.append(f"t-statistic: {t_stat:.3f}")
    output.append(f"p-value: {p_value:.6f}")
    output.append("")

    output.append(interpretation)
    output.append("")

    output.append(f"Western Europe mean happiness: {europe_mean:.3f}")
    output.append(f"Sub-Saharan Africa mean happiness: {africa_mean:.3f}")
    output.append(f"Region comparison t-statistic: {t_stat2:.3f}")
    output.append(f"Region comparison p-value: {p_value2:.6f}")

    with open("personal_output/hypothesis_test_results.txt", "w") as f:
        f.write("\n".join(output))
    return {
        "mean_2019": mean_2019,
        "mean_2020": mean_2020,
        "t_statistic": t_stat,
        "p_value": p_value,
        "interpretation": interpretation
    }

@task
def mult_corr_problem(file):
    df = pd.read_csv(file, sep=";", decimal=",")
    logger = get_run_logger()

    exclude_cols = ["Happiness score", "Ranking"]

    numeric_cols = df.select_dtypes(include="number").columns
    test_cols = [c for c in numeric_cols if c not in exclude_cols]

    number_of_tests = len(test_cols)
    adjusted_alpha = 0.05 / number_of_tests



    output = []
    correlations = []
    output.append(f"Bonferroni-adjusted alpha = {adjusted_alpha:.6f}\n")

    logger.info(f"Bonferroni-adjusted alpha = {adjusted_alpha:.6f}")

    for col in test_cols:
        r, p = pearsonr(df[col], df['Happiness score'])

        significant = p < adjusted_alpha
        correlations.append({
            "variable": col,
            "correlation": r,
            "p_value": p,
            "significant": significant
        })

        line = (f"{col}: r={r:.3f}, p={p:.6f},significant={significant}"
        )

        logger.info(line)
        output.append(line)




    with open("personal_output/correlation_tests.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    


    
    significant_corrs = [c for c in correlations if c["significant"]]

    if significant_corrs:
        strongest = max(significant_corrs,key=lambda x: abs(x["correlation"]))

        strongest_result = (
            f"The variable most strongly correlated with happiness score "
            f"after Bonferroni correction is {strongest['variable']} "
            f"with a Pearson correlation of r={strongest['correlation']:.3f}."
        )

    else:
        strongest_result = ("No variables showed a statistically significant correlation with happiness score after Bonferroni correction."
        )
    return {"strongest_correlation": strongest_result}

def summaryR():
    logger = get_run_logger()

    output = []
    output.append("World Happiness Report Analysis Summary\n")

    # Load and merge data
    load_result = load_Data()
    logger.info(f"Dataset summary:\n{load_result}")
    output.append(f"Dataset summary:\n{load_result}\n")

    file = os.path.join("./outputs", "WHOutput.csv")

    # Descriptive statistics
    des_result = des_stats(file)

    des_output = (
        "Descriptive statistics summary:\n"
        f"Top 3 regions by mean happiness:\n"
        f"{des_result['top_3_regions'].to_string()}\n\n"
        f"Bottom 3 regions by mean happiness:\n"
        f"{des_result['bottom_3_regions'].to_string()}\n"
    )

    logger.info(des_output)
    output.append(des_output)

    # Visualizations
    visualizations(file)
    logger.info("Visualizations completed successfully")
    output.append("Visualizations completed successfully\n")

    # Hypothesis testing
    hypothesis_result = hypothesis_tests(file)

    hypothesis_output = (
        f"Hypothesis test results:\n"
        f"2019 mean: {hypothesis_result['mean_2019']:.3f}\n"
        f"2020 mean: {hypothesis_result['mean_2020']:.3f}\n"
        f"T-statistic: {hypothesis_result['t_statistic']:.3f}\n"
        f"P-value: {hypothesis_result['p_value']:.6f}\n"
        f"Interpretation: {hypothesis_result['interpretation']}\n"
    )

    logger.info(hypothesis_output)
    output.append(hypothesis_output)

    # Correlation analysis
    corr_result = mult_corr_problem(file)

    correlation_output = (
        f"Correlation analysis result:\n"
        f"{corr_result['strongest_correlation']}\n"
    )

    logger.info(correlation_output)
    output.append(correlation_output)

    with open("outputs/analysis_summary.txt","w",) as f:
        f.write("\n".join(output))
@flow
def happiness_pipeline():
    summaryR()
   



  
if __name__ == "__main__":
     happiness_pipeline()

import pandas as pd
from pathlib import Path
import  matplotlib.pyplot as plt

import numpy as np
import seaborn as sns


problem_path = Path("/Volumes/UBUNTU18/USC/IBIS-EEG/derivatives/aep_events/tmp_to_fix")

problem_files = list(problem_path.glob("*.csv"))
problem_files = [f for f in problem_files if not f.name.startswith(".")]
good_files = list(problem_path.parent.glob("PHI*.csv"))

palette = sns.color_palette()
fig, ax = plt.subplots(constrained_layout=True)

files = problem_files + good_files
files = [f for f in files if not f.name.startswith(".")]
latencies = {"good": [], "problem": []}
for ii, file in enumerate(files, 1):
    if file.name[:7] == "PHI7059":
        # Skip this file, it is invalid
        continue
    df = pd.read_csv(file)
    tone_trigger_idxs = df[df["description"] == "ton+"].index
    # There can be some BAD_ annotations in between the tone triggers and the DIN1 events
    tone_onset_idxs = df.loc[(df["description"] == "DIN1") & (df["description"].shift(1) == "ton+")].index
    df_tone_triggers = df.iloc[tone_trigger_idxs]
    df_tone_onsets = df.iloc[tone_onset_idxs]
    if file in problem_files:
        latencies["problem"].append(df_tone_onsets["seconds_after_previous_event"].mean())
    else:
        latencies["good"].append(df_tone_onsets["seconds_after_previous_event"].mean())
    assert df_tone_onsets["description"].unique() == "DIN1", f"Expected DIN1 events, found {df_tone_onsets.unique()['description']}"
    assert df_tone_triggers["description"].unique() == "ton+", f"Expected ton+ events, found {df_tone_triggers.unique()['description']}"

    triggers_to_plot = df_tone_triggers["onset"].values[:10]
    onsets_to_plot = df_tone_onsets["onset"].values[:10]
    distances_to_plot = onsets_to_plot - triggers_to_plot
    triggers_to_plot = triggers_to_plot - triggers_to_plot[0]
    

    color_1 = palette[0] if file.parent.name == "tmp_to_fix" else palette[2]
    color_2 = palette[1] if file.parent.name == "tmp_to_fix" else palette[3]
    alpha = 1 if file.parent.name == "tmp_to_fix" else 0.5
    ax.plot(triggers_to_plot, [ii] * len(triggers_to_plot), "o", label="ton+", color=color_1, marker=">")
    ax.plot(triggers_to_plot + distances_to_plot, [ii] * len(distances_to_plot), "o", label="DIN1", color=color_2, alpha=alpha)
    ax.set_title("ton+ and DIN1 events")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("File index")
    ax.set_yticks(range(1, len(files) + 1), labels=[f.name[:7] for f in files])


   
ax.axhline(y=(len(problem_files) + 1), color=palette[4], linestyle="--")
fig.show()
fig.savefig(good_files[0].parent / "logs" / "ton_plus_din1_events.png")

df_violin = pd.concat(
    [
        pd.DataFrame({"latency": vals, "group": [key] * len(vals)})
        for key, vals
        in latencies.items()
    ]
)
df_violin["latency"] = df_violin["latency"] * 1000 # convert to ms
fig_2, ax_2 = plt.subplots(constrained_layout=True)
sns.violinplot(data=df_violin, x="group", y="latency", ax=ax_2)
ax_2.set_ylabel("Latency (ms)")
ax_2.set_xlabel("Group")
ax_2.set_title("Average Latency of DIN1 events after ton+ events per participant")
fig_2.savefig(good_files[0].parent / "logs" / "latency_violin.png")
fig_2.show()
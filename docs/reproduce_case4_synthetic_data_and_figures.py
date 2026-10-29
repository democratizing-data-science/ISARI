#!/usr/bin/env python3
"""Reproduce the synthetic Case 4 dataset and all six teaching figures.

All records and values are fictional. They are generated only to demonstrate
an offline, privacy-by-design qualitative research workflow. The outputs must
not be interpreted as findings about any real population.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
SEED = 42
rng = np.random.default_rng(SEED)
COLORS = {
    "navy": "#001F5B", "pink": "#E4007C", "teal": "#0E9F9A",
    "red": "#990000", "gold": "#F2A900", "dark": "#172033"
}
GROUPS = ["Housing insecurity", "Disability services",
          "Recent migration", "Care-experienced youth"]

# Synthetic participant-level teaching data.
rows = []
for group in GROUPS:
    for i in range(9):
        rows.append({
            "group": group,
            "case_id": f"{group[:2].upper()}{i+1:02d}",
            "turns": int(rng.integers(14, 31)),
            "sensitivity": int(rng.integers(3, 6)),
            "distinctive_context": int(rng.integers(2, 6)),
            "model_memo_risk": int(rng.integers(1, 5)),
        })
df = pd.DataFrame(rows)
df.to_csv(OUT / "case4_synthetic_participants.csv", index=False)

# Figure 1: cohort.
counts = df.groupby("group", sort=False).size()
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(counts.index, counts.values,
              color=[COLORS["navy"], COLORS["pink"], COLORS["teal"], COLORS["red"]])
ax.set_ylim(0, 11); ax.set_ylabel("Synthetic participants")
ax.set_title("Synthetic sensitive-evidence teaching corpus", loc="left",
             fontweight="bold", color=COLORS["navy"])
ax.tick_params(axis="x", rotation=15); ax.spines[["top", "right"]].set_visible(False)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.2,
            str(int(bar.get_height())), ha="center", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "case4_fig1_cohort.png", dpi=220); plt.close(fig)

# Figure 2: artifact risk matrix.
artifacts = ["Original audio", "Reviewed transcript", "Pseudonymized excerpt",
             "Embedding/feature file", "Local model memo", "De-identified figure"]
M = np.array([[5,5,1],[4,5,2],[3,4,3],[3,2,4],[3,4,4],[1,2,5]])
fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(M, cmap="RdYlGn_r", vmin=1, vmax=5, aspect="auto")
ax.set_xticks(range(3), ["Re-identification risk", "Context sensitivity", "Export risk"])
ax.set_yticks(range(len(artifacts)), artifacts)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, str(M[i,j]), ha="center", va="center", fontweight="bold",
                color="white" if M[i,j] >= 4 else COLORS["dark"])
ax.set_title("Derivative artifacts remain sensitive", loc="left",
             fontweight="bold", color=COLORS["navy"])
fig.colorbar(im, ax=ax, label="Illustrative risk score (1–5)")
fig.tight_layout(); fig.savefig(OUT / "case4_fig2_artifact_risk.png", dpi=220); plt.close(fig)

# Figure 3: data-minimization funnel. This replacement avoids the prior blank render.
stages = ["Raw recordings", "Locally reviewed records", "Pseudonymized analytical set",
          "Bounded prompt packet", "Report examples"]
values = [36, 36, 36, 12, 4]
widths = [0.92, 0.82, 0.72, 0.42, 0.24]
colors = [COLORS["red"], COLORS["pink"], COLORS["gold"], COLORS["teal"], COLORS["navy"]]
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 1); ax.set_ylim(0, 6); ax.axis("off")
for idx, (stage, value, width, color) in enumerate(zip(stages, values, widths, colors)):
    y = 5.25 - idx * 1.02
    x = (1-width)/2
    box = FancyBboxPatch((x, y), width, .68, boxstyle="round,pad=.02,rounding_size=.03",
                         facecolor=color, edgecolor="none")
    ax.add_patch(box)
    ax.text(.5, y+.42, stage, ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")
    ax.text(.5, y+.15, f"{value} synthetic records", ha="center", va="center",
            color="white", fontsize=10)
    if idx < len(stages)-1:
        ax.annotate("", xy=(.5, y-.25), xytext=(.5, y-.02),
                    arrowprops=dict(arrowstyle="->", color="#667085", lw=1.6))
ax.set_title("Data minimization across the local workflow", loc="left",
             fontsize=17, fontweight="bold", color=COLORS["navy"], pad=12)
ax.text(.5, .15, "Synthetic teaching values; no real participant data are represented.",
        ha="center", color=COLORS["dark"], fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "case4_fig3_minimization.png", dpi=220,
                                facecolor="white", bbox_inches="tight"); plt.close(fig)

# Figure 4: researcher dispositions.
accepted=np.array([8,7,8,6]); revised=np.array([4,5,3,5]); rejected=np.array([2,3,2,4])
x=np.arange(4); w=.24
fig,ax=plt.subplots(figsize=(10,5))
ax.bar(x-w, accepted, w, label="Retained after verification", color=COLORS["teal"])
ax.bar(x, revised, w, label="Revised", color=COLORS["gold"])
ax.bar(x+w, rejected, w, label="Rejected", color=COLORS["red"])
ax.set_xticks(x, GROUPS, rotation=15); ax.set_ylabel("Synthetic AI-assisted observations")
ax.set_title("Researcher dispositions by vulnerable-population context", loc="left",
             fontweight="bold", color=COLORS["navy"])
ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "case4_fig4_dispositions.png", dpi=220); plt.close(fig)

# Figure 5: compression safeguards.
cats=["Frequency visibility","Context preservation","Contradiction visibility",
      "Group balance","Source traceability"]
before=[4.6,2.3,1.9,2.6,2.1]; after=[4.1,4.5,4.2,4.4,4.8]
angles=np.linspace(0,2*np.pi,len(cats),endpoint=False).tolist(); angles += angles[:1]
fig=plt.figure(figsize=(7,7)); ax=plt.subplot(111, polar=True)
ax.plot(angles,before+before[:1],color=COLORS["red"],lw=2,label="Unbounded summary")
ax.fill(angles,before+before[:1],color=COLORS["red"],alpha=.12)
ax.plot(angles,after+after[:1],color=COLORS["teal"],lw=2,label="Evidence-grounded workflow")
ax.fill(angles,after+after[:1],color=COLORS["teal"],alpha=.12)
ax.set_xticks(angles[:-1],cats); ax.set_ylim(0,5)
ax.set_title("Safeguards against minority-account compression", fontweight="bold",
             color=COLORS["navy"], pad=25)
ax.legend(loc="lower center",bbox_to_anchor=(.5,-.2),frameon=False)
fig.tight_layout(); fig.savefig(OUT / "case4_fig5_compression.png", dpi=220); plt.close(fig)

# Figure 6: claim-authorization trail.
fig,ax=plt.subplots(figsize=(11,4.5)); ax.axis("off")
labels=["Source segment\nS-21","Deterministic output\nD-08","Local memo\nM-14",
        "Researcher review\nREVISE","Final claim\nC-03"]
xs=np.linspace(.08,.92,5); ys=[.55,.72,.42,.72,.55]
colors=[COLORS["navy"],COLORS["teal"],COLORS["pink"],COLORS["gold"],COLORS["red"]]
for x0,y0,label,color in zip(xs,ys,labels,colors):
    ax.text(x0,y0,label,ha="center",va="center",color="white",fontsize=10,
            bbox=dict(boxstyle="round,pad=.6",fc=color,ec="none"))
for i in range(4):
    ax.annotate("",xy=(xs[i+1]-.08,ys[i+1]),xytext=(xs[i]+.08,ys[i]),
                arrowprops=dict(arrowstyle="->",color="#667085",lw=1.7))
ax.text(.5,.92,"Sensitive-claim authorization trail",ha="center",fontsize=18,
        fontweight="bold",color=COLORS["navy"])
ax.text(.5,.12,"Every retained claim preserves source, transformation, model, researcher decision, and uncertainty.",
        ha="center",fontsize=11,color=COLORS["dark"])
fig.savefig(OUT / "case4_fig6_claim_audit.png", dpi=220, bbox_inches="tight"); plt.close(fig)

print(f"Created {len(df)} synthetic records with seed {SEED} in {OUT}")

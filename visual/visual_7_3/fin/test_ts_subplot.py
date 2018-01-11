
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



import matplotlib.dates as mdates

dates = pd.date_range("2013", periods=12, freq='MS')

fig, ax = plt.subplots()
ax.plot(dates, np.array([2,5,4,1,7,6,7,8,1,5,9,3]), "-")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
plt.show()


"""
pd.__version__
'0.21.0'
"""
import pandas.plotting._converter as pandacnv
pandacnv.register()

dates = pd.date_range("2013", periods=12, freq='MS')
x = np.array([2,5,4,1,7,6,7,8,1,5,9,3])
x_u = x + 1
x_l = x - 1
y = np.random.rand(12)*10
z = np.arange(10,-2,-1)
fig, axes = plt.subplots(3, 1)
axes[0].plot(dates, x, "-", color="k")
axes[0].fill_between(dates, x_u, x_l, facecolor='lightskyblue', alpha=0.3, interpolate=True)
#axes[0].fill_between(dates.to_pydatetime(), x_u, x_l, facecolor='lightskyblue', alpha=0.3, interpolate=True)
axes[0].set_ylim([0, 12])
axes[0].set_yticks([0, 4, 6, 8, 12])
#plt.setp(axes[0].get_xticklabels(), visible=False)
str_year = "2013"
#axes[0].set_title('A_30, '+str_year)
axes[0].set_xlabel('month')
axes[0].set_ylabel('A_30')
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m'))
#axes[1].set_title('theta_30, '+str_year)
axes[1].plot(dates, y)
axes[2].plot(dates, z)
plt.tight_layout()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
for ax in axes:
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))





df = pd.DataFrame({'C1': {'a': 1,'b': 15,'c': 9,'d': 7,'e': 2,'f': 2,'g': 6,'h': 5,'k': 5,'l': 8},
          'C2': {'a': 6,'b': 18,'c': 13,'d': 8,'e': 6,'f': 6,'g': 8,'h': 9,'k': 13,'l': 15}})

JG1 = sns.jointplot("C1", "C2", data=df, kind='reg')
JG2 = sns.jointplot("C1", "C2", data=df, kind='kde')

#subplots migration

f = plt.figure()
for J in [JG1, JG2]:
    for A in J.fig.axes:
        f._axstack.add(f._make_key(A), A)

#subplots size adjustment

f.axes[0].set_position([0.05, 0.05, 0.4,  0.4])
f.axes[1].set_position([0.05, 0.45, 0.4,  0.05])
f.axes[2].set_position([0.45, 0.05, 0.05, 0.4])
f.axes[3].set_position([0.55, 0.05, 0.4,  0.4])
f.axes[4].set_position([0.55, 0.45, 0.4,  0.05])
f.axes[5].set_position([0.95, 0.05, 0.05, 0.4])



df = pd.DataFrame({'C1': {'a': 1,'b': 15,'c': 9,'d': 7,'e': 2,'f': 2,'g': 6,'h': 5,'k': 5,'l': 8},
          'C2': {'a': 6,'b': 18,'c': 13,'d': 8,'e': 6,'f': 6,'g': 8,'h': 9,'k': 13,'l': 15}})

#g = sns.JointGrid('C1', 'C2', df)
#g.plot_joint(sns.jointplot)

fig, axes = plt.subplots(1, 2)
sns.jointplot("C1", "C2", data=df, kind='reg', annot_kws=dict(stat="r"), ax=axes[0])
sns.jointplot("C1", "C2", data=df, kind='kde', ax=axes[1])

#JG1 = sns.jointplot("C1", "C2", data=df, kind='reg', annot_kws=dict(stat="r"), ax=axes[0])
#JG2 = sns.jointplot("C1", "C2", data=df, kind='kde', ax=axes[1])

axes[0] = JG1.ax_joint
axes[1] = JG2.ax_joint

fig, axes = plt.subplots(1, 2)
g = sns.JointGrid('C1', 'C2', df)
g.annot_kws=dict(stat="r")
#g = sns.JointGrid('C1', 'C2', df)
g.plot_joint(sns.regplot, ax=axes[0])
g.plot_joint(sns.distplot, ax=axes[0])





f, (ax1, ax2) = plt.subplots(2)
sns.jointplot("C1", "C2", data=df, kind='reg', annot_kws=dict(stat="r"), ax=ax1)
sns.jointplot("C1", "C2", data=df, kind='kde', ax=ax2)
sns.regplot("C1", "C2", data=df, ax=ax1)
sns.kdeplot("C1", "C2", data=df, ax=ax2)










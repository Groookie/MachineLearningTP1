import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot([1,2,3,4,5], [10,5,10,5,10], 'r-')
ax2 = fig.add_subplot(222)
ax2.plot([1,2,3,4], [1,4,9,16], 'k-')
ax3 = fig.add_subplot(223)
ax3.plot([1,2,3,4], [1,10,100,1000], 'b-')
ax4 = fig.add_subplot(224)
ax4.plot([1,2,3,4], [0,0,1,1], 'g-')
plt.tight_layout()
fig = plt.gcf()
plotly_fig = tls.mpl_to_plotly( fig )
plotly_fig['layout']['title'] = 'Simple Subplot Example Title'
plotly_fig['layout']['margin'].update({'t':40})
py.iplot(plotly_fig)
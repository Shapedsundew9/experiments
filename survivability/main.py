from landscape import mountain_cluster, mountain_domain
from population import population
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from numpy import array, isfinite, argsort, min, max, NaN, log
from numpy.random import choice


l = mountain_domain()
p = population(cr=l.x_max, n=1000)
#l = mountain_cluster()
#p = population(cx=l.x_min * 0.9, cy=l.y_min * 0.9)

def ff(i, p):
    # FIXME: Calculate fitness at the population layer.
    if i.x < l.x_min or i.x > l.x_max or i.y < l.y_min or i.y > l.y_max:
        i.f = NaN
    else:
        i.f = -l.function(i.x, i.y)
    return i

def sf(p):
    # FIXME: Treat survivability at the population layer.
    if len(p) > p.n:
        s = array([i.f for i in p])
        idx = argsort(s)
        top = int(p.n/4)
        best_idx = idx[-top:]
        best = [p[i] for i in best_idx]
        s = s - s.min()
        s[~isfinite(s)] = 0.0
        s[best_idx] = 0.0
        sum_s = s.sum()
        p.done = sum_s < p.e
        s = [1.0 / len(s)] * len(s) if p.done else s / sum_s
        rest = choice(p, p.n - top, replace=False, p=s)
        best.extend(rest)
        return best
    return p

def sf_gradient(p):
    # FIXME: Treat survivability at the population layer.
    if len(p) > p.n:
        s = array([i.f for i in p])
        idx = argsort(s)
        top = int(p.n/4)
        best_idx = idx[-top:]
        best = [p[i] for i in best_idx]
        s = s - s.min()
        s[~isfinite(s)] = 0.0
        s[best_idx] = 0.0
        sum_s = s.sum()
        p.done = sum_s < p.e
        s = [1.0 / len(s)] * len(s) if p.done else s / sum_s
        rest = choice(p, p.n - top, replace=False, p=s)
        best.extend(rest)
        return best
    return p

p.set_ff(ff)
p.set_sf(sf)
show = True

sz = 8 if show else 20
fig, axs = plt.subplot_mosaic([['map','map', 'fitness'], ['map', 'map', 'generation']], figsize=(int(sz * 1.75), sz))

ax1 = axs['map']
tt = ax1.set_title(f'Generation {p.g}: Max Fitness {p.bf}', animated=True)
cf = ax1.contourf(l.xx, l.yy, l.zz, cmap='terrain')
pp = ax1.scatter([i.x for i in p], [i.y for i in p], marker='o', color='red', linewidth=0, animated=True)
cb = fig.colorbar(cf, ax=ax1)

ax2 = axs['fitness']
(bf, ) = ax2.plot(p.bg, p.bf, color='green', animated=True)
(wf, ) = ax2.plot(p.bg, p.wf, color='red', animated=True)
(mf, ) = ax2.plot(p.bg, p.mf, color='black', linestyle=':', animated=True)
ax2.set_xlim((0, 500))
ax2.set_ylim((0, max(l.zz) * 1.1))
ax2.set_title('Live Population Fitness')
ax2.set_xlabel('Generation')
ax2.set_ylabel('Fitness')

ax3 = axs['generation']
(lg, ) = ax3.plot(p.bg, array(p.bg) - array(p.lg), color='red', animated=True)
(mg, ) = ax3.plot(p.bg, array(p.bg) - array(p.mg), color='black', linestyle=':', animated=True)
ax3.set_xlim((0, 500))
ax3.set_ylim((0, 200))
#ax3.set_yscale('log')
ax3.set_title('Live Population Generations')
ax3.set_xlabel('Generation')
ax3.set_ylabel('Relative Generation')

plt.show(block=False)
bg = fig.canvas.copy_from_bbox(fig.bbox)
ax1.draw_artist(pp)
ax1.draw_artist(tt)

ax2.draw_artist(bf)
ax2.draw_artist(wf)
ax2.draw_artist(mf)

ax3.draw_artist(lg)
ax3.draw_artist(mg)

fig.canvas.blit(fig.bbox)

while not p.breed():
    fig.canvas.restore_region(bg)
    tt.set_text(f'Generation {p.g}: Max Fitness {p.bf[-1]:0.3}')
    pp.set_offsets(tuple(zip([i.x for i in p], [i.y for i in p])))
    ax1.draw_artist(tt)
    ax1.draw_artist(pp)

    bf.set_xdata(p.bg)
    bf.set_ydata(p.bf)
    ax2.draw_artist(bf)
    wf.set_xdata(p.bg)
    wf.set_ydata(p.wf)
    ax2.draw_artist(wf)
    mf.set_xdata(p.bg)
    mf.set_ydata(p.mf)
    ax2.draw_artist(mf)

    lg.set_xdata(p.bg)
    lg.set_ydata(array(p.bg) - array(p.lg))
    ax3.draw_artist(lg)
    mg.set_xdata(p.bg)
    mg.set_ydata(array(p.bg) - array(p.mg))
    ax3.draw_artist(mg)

    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()

x=input('Done: Hit Enter')
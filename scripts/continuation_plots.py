"""Publication layouts adapted from the user's workflow_1d.py/workflow_2d.py.

1D: four-component overview and seagrass + absent-seagrass strip.
Depth comparisons and 2D maps: 1, 2 or 4 panels, side legends, shared axes.
Every figure is saved as app PNG and vector PDF.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError
from model.parameter_metadata import parameter_schema

VARIABLE_COLORS={'M':'#e08b24','S':'#278360','P_area':'#3c72bf','N_area':'#795eb5'}
STATE_COLORS={'MSP':'#8055b5','MS':'#9e9d41','MP':'#ac476e','SP':'#228b86',
              'M':'#df8531','S':'#258c58','P':'#418cc8','0':'#777777'}
BASE_COLORS={'MSP':'#8062a5','MS':'#aaa14d','MP':'#b86584','SP':'#40948b',
             'M':'#e4a064','S':'#73a985','P':'#84aed5','0':'#a5a5a5','?':'#eeeeee','∅':'#f7f7f7'}
OTHER_COLORS=['#cabd9b','#d09b9d','#8caa75','#9ba5bc','#c3a3ca','#adc3c3']
STYLES={'stable':'-','unstable':(0,(1,2)),'nonhyperbolic':'--'}
EVENT_STYLES={'fold':('-',2.,'#151515','Fold'), 'hopf':((0,(6,3)),1.9,'#bc244e','Hopf'),
              'invasion':((0,(1.3,2.3)),1.7,'#111111','Invasion')}


def axis_label(axis):
    metadata=parameter_schema()[axis.name]
    return f'{axis.name} [{metadata["unit"]}]'


def save(fig,folder,name):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f'{name}.png',dpi=270,bbox_inches='tight')
    fig.savefig(folder/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def segments(path):
    if not path:return
    start=0
    for i in range(1,len(path)):
        if path[i]['stability']!=path[i-1]['stability']:
            yield path[start:i+1],path[i-1]['stability']
            start=i
    yield path[start:],path[-1]['stability']


def draw_branches(ax,dataset,component,community_colors=False,zero_strip=False):
    positions={'0':0,'M':1,'P':2,'MP':3}
    if not zero_strip:draw_periodic_envelopes(ax,dataset,component,community_colors)
    for path in dataset['paths']:
        if not path:continue
        community=path[0]['community']
        if zero_strip and community not in positions:continue
        if community_colors and not zero_strip and 'S' not in community:continue
        for rows,stability in segments(path):
            color=STATE_COLORS[community] if community_colors else VARIABLE_COLORS[component]
            ys=[positions[community]]*len(rows) if zero_strip else [r[component] for r in rows]
            ax.plot([r['x'] for r in rows],ys,color=color,lw=2.1,ls=STYLES[stability],
                    marker='.' if len(rows)==1 else None,markersize=3)
    for event in dataset.get('events',[]):
        if zero_strip and event['community'] not in positions:continue
        if community_colors and not zero_strip and 'S' not in event['community']:continue
        marker={'fold':'^','hopf':'D','invasion':'o'}[event['kind']]
        value=positions[event['community']] if zero_strip else event[component]
        ax.scatter(event['x'],value,s=35,marker=marker,facecolors='white',
                   edgecolors=EVENT_STYLES[event['kind']][2],zorder=8)
    ax.grid(alpha=.13)




# Colors describe attractor combinations, not species communities.
ATTRACTOR_COLORS = {
    'stable_equilibrium_cycle_not_confirmed': '#278360',
    'stable_cycle_no_stable_equilibrium_found': '#df8531',
    'coexistence': '#8055b5',
    'stable_cycle_equilibria_unresolved': '#b69c68',
    'unresolved': '#999999',
}

ATTRACTOR_LABELS = {
    'stable_equilibrium_cycle_not_confirmed':
        'Equilibrium; cycle not confirmed',

    'stable_cycle_no_stable_equilibrium_found':
        'Cycle; no stable equilibrium found',

    'coexistence':
        'Equilibrium + periodic attractor',

    'stable_cycle_equilibria_unresolved':
        'Cycle; equilibrium search unresolved',

    'unresolved':
        'Unresolved',
}


def is_sampled_analysis(data):
    """Distinguish shooting results from native periodic branches."""

    payload = data.get('periodic_analysis', {})

    return (
        payload.get('method')
        == 'Adaptive ODE integration and shooting'
        or 'sampling_coordinates' in payload
    )


def sampled_locations(data):
    """Return the complete planned grid, including unvisited points.

    Keeping unsuccessful and unvisited locations in the sequence
    prevents the plotting code from bridging missing results.
    """

    payload = data.get('periodic_analysis', {})

    locations = {}

    for row in payload.get('locations', []):

        if 'u' not in row:
            continue

        u = round(float(row['u']), 10)

        locations[u] = row

    coordinates = set(locations)

    for value in payload.get('sampling_coordinates', []):

        coordinates.add(
            round(float(value), 10)
        )

    # Compatibility with previously saved shooting calculations.
    for value in payload.get(
        'coverage', {}
    ).get('unvisited_locations', []):

        if isinstance(value, (list, tuple)) and value:

            coordinates.add(
                round(float(value[0]), 10)
            )

    return [
        (u, locations.get(u))
        for u in sorted(coordinates)
    ]


def verified_sample(row):
    """Only accept numerically verified stable shooting solutions."""

    if row is None:
        return False

    if not row.get('stable_cycle_found', False):
        return False

    if row.get('status') != 'verified_periodic':
        return False

    if row.get('stability') != 'stable':
        return False

    required = ['period', 'x']

    for component in VARIABLE_COLORS:

        required.extend([
            component+'_min',
            component+'_max',
            component+'_mean',
        ])

    try:

        if not all(
            np.isfinite(float(row[key]))
            for key in required
        ):
            return False

        if float(row['period']) <= 0:
            return False

        for component in VARIABLE_COLORS:

            lo = float(row[component+'_min'])
            hi = float(row[component+'_max'])

            if lo > hi:
                return False

    except (KeyError, TypeError, ValueError):

        return False

    return True


def sampled_neighbors(a, b, data):
    """Check that two samples are sufficiently close in parameter space."""

    if a is None or b is None:
        return False

    gap = abs(
        float(a['u'])-float(b['u'])
    )

    resolution = data.get(
        'periodic_analysis', {}
    ).get(
        'settings', {}
    ).get('scan_resolution')

    if resolution is None:

        # Conservative fallback for older shooting results.
        return gap <= .01+1e-9

    resolution = max(1, int(resolution))

    return (
        0 < gap <= 1/resolution+1e-9
    )


def compatible_sampled_cycles(a, b, data):
    """Determine whether visual interpolation is appropriate.

    This is NOT a mathematical proof of periodic-branch connectivity.

    Requirements:
        Both cycles are independently verified and stable.
        The parameter locations are adjacent and sufficiently close.
        One solution was initialized directly from the other.
        Their periods and biomass ranges are reasonably similar.
    """

    if not verified_sample(a):
        return False

    if not verified_sample(b):
        return False

    if not sampled_neighbors(a, b, data):
        return False

    ua = float(a['u'])
    ub = float(b['u'])

    parent_a = a.get('cycle_parent_u')
    parent_b = b.get('cycle_parent_u')

    linked = (
        parent_a is not None
        and abs(float(parent_a)-ub) <= 1e-9
    ) or (
        parent_b is not None
        and abs(float(parent_b)-ua) <= 1e-9
    )

    if not linked:
        return False

    # Avoid visually connecting substantially different periods.
    period_change = abs(
        np.log(float(a['period'])/float(b['period']))
    )

    if period_change > .30:
        return False

    # Compare the full biomass envelopes, including nutrients.
    # Logarithmic scaling prevents large biomass components from
    # dominating the similarity check.
    for component in VARIABLE_COLORS:

        for bound in ('min', 'max', 'mean'):

            key = component+'_'+bound

            va = max(0., float(a[key]))
            vb = max(0., float(b[key]))

            difference = abs(
                np.log1p(va)-np.log1p(vb)
            )

            if difference > .40:
                return False

    return True


def draw_sampled_envelopes(
    ax, data, component, community_colors=False
):
    """Draw independently verified periodic biomass ranges."""

    color = (
        STATE_COLORS['MSP']
        if community_colors
        else VARIABLE_COLORS[component]
    )

    locations = sampled_locations(data)

    # Draw shaded envelopes only between adjacent, compatible cycles.
    for (_, a), (_, b) in zip(
        locations, locations[1:]
    ):

        if not compatible_sampled_cycles(a, b, data):
            continue

        x = [
            float(a['x']),
            float(b['x']),
        ]

        minima = [
            float(a[component+'_min']),
            float(b[component+'_min']),
        ]

        maxima = [
            float(a[component+'_max']),
            float(b[component+'_max']),
        ]

        ax.fill_between(
            x,
            minima,
            maxima,
            color=color,
            alpha=.25,
            lw=0,
            zorder=1,
        )

        # Subtle boundaries make the periodic range readable.
        ax.plot(
            x,
            minima,
            color=color,
            lw=.75,
            alpha=.65,
            zorder=2,
        )

        ax.plot(
            x,
            maxima,
            color=color,
            lw=.75,
            alpha=.65,
            zorder=2,
        )

    # Every verified solution remains visible, including isolated
    # solutions that could not safely be connected to a neighbor.
    for _, row in locations:

        if not verified_sample(row):
            continue

        x = float(row['x'])

        ax.vlines(
            x,
            float(row[component+'_min']),
            float(row[component+'_max']),
            color=color,
            alpha=.65,
            lw=1.,
            zorder=3,
        )

        # Small mean marker identifies the actual sampled location.
        ax.scatter(
            x,
            float(row[component+'_mean']),
            color=color,
            s=5,
            alpha=.65,
            zorder=3,
        )


def attractor_category(row):
    """Return a defensible attractor classification for the plot."""

    if row is None:
        return 'unresolved'

    stable_cycle = verified_sample(row)

    stable_equilibrium = (
        int(row.get('stable_equilibrium_count', 0)) > 0
        or bool(row.get('stable_equilibria', ''))
    )

    equilibrium_error = bool(
        row.get('equilibrium_search_error', '')
    )

    if stable_cycle and stable_equilibrium:

        return 'coexistence'

    if stable_cycle and equilibrium_error:

        return 'stable_cycle_equilibria_unresolved'

    if stable_cycle:

        return 'stable_cycle_no_stable_equilibrium_found'

    if stable_equilibrium:

        return 'stable_equilibrium_cycle_not_confirmed'

    return 'unresolved'


def draw_attractor_strip(ax, data, axis):
    """Show the attractor classification at tested parameter values.

    Colors between adjacent samples are visual interpolation only.
    Missing or unsuccessful searches are not interpreted as proven
    absence of oscillatory attractors.
    """

    locations = sampled_locations(data)

    for (ua, a), (ub, b) in zip(
        locations, locations[1:]
    ):

        if a is None or b is None:
            continue

        if not sampled_neighbors(a, b, data):
            continue

        category_a = attractor_category(a)
        category_b = attractor_category(b)

        if category_a != category_b:
            continue

        if category_a == 'unresolved':
            continue

        if category_a in (
            'coexistence',
            'stable_cycle_no_stable_equilibrium_found',
            'stable_cycle_equilibria_unresolved',
        ):

            if not compatible_sampled_cycles(a, b, data):
                continue

        elif category_a == (
            'stable_equilibrium_cycle_not_confirmed'
        ):

            # Avoid joining different equilibrium communities.
            communities_a = set(
                a.get('stable_equilibria', '').split('+')
            )-{''}

            communities_b = set(
                b.get('stable_equilibria', '').split('+')
            )-{''}

            if not communities_a.intersection(communities_b):
                continue

        ax.axvspan(
            axis.physical(ua),
            axis.physical(ub),
            facecolor=ATTRACTOR_COLORS[category_a],
            alpha=.45,
            edgecolor='none',
            zorder=1,
        )

    # Mark actual completed tests. Unvisited locations remain blank.
    for u, row in locations:

        if row is None:
            continue

        category = attractor_category(row)

        ax.scatter(
            axis.physical(u),
            .5,
            marker='s',
            s=10,
            color=ATTRACTOR_COLORS[category],
            edgecolors='none',
            zorder=3,
        )

    ax.set_xlim(axis.low, axis.high)
    ax.set_ylim(0., 1.)

    ax.set_yticks([])
    ax.set_ylabel(
        'Attractors',
        fontsize=8,
    )

    ax.grid(False)

    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def attractor_handles(datasets):
    """Conditional legend: only show classifications actually present."""

    used = {
        attractor_category(row)
        for data in datasets
        if is_sampled_analysis(data)
        for _, row in sampled_locations(data)
        if row is not None
    }

    return [
        Patch(
            facecolor=ATTRACTOR_COLORS[key],
            alpha=.65,
            label=ATTRACTOR_LABELS[key],
        )
        for key in ATTRACTOR_COLORS
        if key in used
    ]



def draw_periodic_envelopes(
    ax, data, component, community_colors=False
):
    """Draw either sampled stable cycles or native periodic branches."""

    if is_sampled_analysis(data):

        draw_sampled_envelopes(
            ax,
            data,
            component,
            community_colors,
        )

        return

    # Original native continuation visualization.
    color = (
        STATE_COLORS['MSP']
        if community_colors
        else VARIABLE_COLORS[component]
    )

    branches = data.get(
        'periodic_analysis', {}
    ).get('branches', [])

    for branch in branches:

        points = branch.get('points', [])

        if not points:
            continue

        first = points[0]

        ax.vlines(
            first['x'],
            first[component+'_min'],
            first[component+'_max'],
            color=color,
            alpha=.5,
            lw=.8,
        )

        for arm in branch.get('arms', []):

            path = [
                first,
                *arm.get('records', []),
            ]

            # Preserve actual continuation order, including folds.
            for a, b in zip(path, path[1:]):

                stable = (
                    a['stability']
                    == b['stability']
                    == 'stable'
                )

                x = [a['x'], b['x']]

                lo = [
                    r[component+'_min']
                    for r in (a, b)
                ]

                hi = [
                    r[component+'_max']
                    for r in (a, b)
                ]

                if stable:

                    ax.fill_between(
                        x,
                        lo,
                        hi,
                        color=color,
                        alpha=.3,
                        lw=0,
                        zorder=1,
                    )

                else:

                    style = (
                        '--'
                        if 'unstable' in (
                            a['stability'],
                            b['stability'],
                        )
                        else ':'
                    )

                    ax.plot(
                        x, lo,
                        color=color,
                        ls=style,
                        lw=1,
                    )

                    ax.plot(
                        x, hi,
                        color=color,
                        ls=style,
                        lw=1,
                    )

                if b.get('event') == 'cycle_fold_bracket':

                    ax.scatter(
                        b['x'],
                        b[component+'_mean'],
                        marker='s',
                        s=20,
                        color=color,
                        zorder=7,
                    )


def periodic_handles(datasets=None):
    """Show only the relevant periodic-orbit legend entries."""

    if datasets is None:

        # Backward compatibility with other callers.
        return [
            Patch(
                facecolor=STATE_COLORS['MSP'],
                alpha=.3,
                label='Stable periodic envelope',
            ),
            Line2D(
                [], [],
                color=STATE_COLORS['MSP'],
                ls='--',
                label='Unstable periodic envelope',
            ),
            Line2D(
                [], [],
                color=STATE_COLORS['MSP'],
                ls=':',
                label='Periodic stability unresolved',
            ),
        ]

    handles = []

    sampled_present = any(
        verified_sample(row)
        for data in datasets
        if is_sampled_analysis(data)
        for _, row in sampled_locations(data)
    )

    if sampled_present:

        handles.append(
            Patch(
                facecolor=STATE_COLORS['MSP'],
                alpha=.25,
                label='Sampled stable periodic range',
            )
        )

    native_stability = {
        row.get('stability')
        for data in datasets
        if not is_sampled_analysis(data)
        for branch in data.get(
            'periodic_analysis', {}
        ).get('branches', [])
        for row in branch.get('points', [])
    }

    if 'stable' in native_stability:

        handles.append(
            Patch(
                facecolor=STATE_COLORS['MSP'],
                alpha=.3,
                label='Stable periodic envelope',
            )
        )

    if 'unstable' in native_stability:

        handles.append(
            Line2D(
                [], [],
                color=STATE_COLORS['MSP'],
                ls='--',
                label='Unstable periodic envelope',
            )
        )

    if native_stability.intersection({
        'nonhyperbolic',
        'unresolved',
    }):

        handles.append(
            Line2D(
                [], [],
                color=STATE_COLORS['MSP'],
                ls=':',
                label='Periodic stability unresolved',
            )
        )

    return handles


def depth_axes(count):
    shape={1:(1,1),2:(1,2),4:(2,2)}[count]
    # Four-depth 2D layout uses the user's 9×4 inch landscape proportions.
    size={1:(6,4),2:(9,3.7),4:(9,4)}[count]
    fig,axes=plt.subplots(*shape,figsize=size,sharex=True,sharey=True,squeeze=False)
    fig.subplots_adjust(left=.12,right=.75,bottom=.16,top=.92,wspace=.30,hspace=.38)
    return fig,axes


def plot_1d(datasets, axis, folder):
    """Original SIMP overview with sampled periodic envelopes."""

    has_attractor_scan = any(
        is_sampled_analysis(data)
        and data.get(
            'periodic_analysis', {}
        ).get('locations')
        for data in datasets
    )

    with plt.rc_context({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 9,
    }):

        # ------------------------------------------------------
        # Four-component biomass overview
        # ------------------------------------------------------

        for data in datasets:

            fig, axes = plt.subplots(
                2,
                2,
                figsize=(12, 8),
                sharex=True,
                layout='constrained',
            )

            titles = (
                'Macroalgae',
                'Seagrass',
                'Phytoplankton',
                'Dissolved nitrogen',
            )

            for ax, component, title in zip(
                axes.flat,
                VARIABLE_COLORS,
                titles,
            ):

                draw_branches(
                    ax, data, component
                )

                ax.set_title(title)

                ax.set_ylabel(
                    component.replace('_area', '')
                    + ' [mmol N m$^{-2}$]'
                )

                ax.set_xlim(
                    axis.low, axis.high
                )

                ax.set_ylim(bottom=0)

            for ax in axes[-1]:

                ax.set_xlabel(
                    axis_label(axis)
                )

            fig.suptitle(
                f'SIMP equilibria and periodic orbits'
                f' · H = {data["H"]:g} m'
            )

            handles = [
                Line2D(
                    [], [],
                    color='black',
                    ls=STYLES[status],
                    label=status.title(),
                )
                for status in (
                    'stable',
                    'unstable',
                    'nonhyperbolic',
                )
            ]

            handles += periodic_handles([data])

            fig.legend(
                handles=handles,
                loc='outside lower center',
                ncol=3,
                frameon=False,
            )

            if is_sampled_analysis(data):

                if data.get('oscillations'):

                    fig.supxlabel(
                        'Shaded periodic ranges interpolate '
                        'between compatible verified samples.',
                        fontsize=8,
                    )

            save(
                fig,
                folder,
                f'equilibrium_biomass_'
                f'{axis.name}_H{data["H"]:g}',
            )

        # ------------------------------------------------------
        # Original seagrass-focused comparison
        # ------------------------------------------------------

        count = len(datasets)

        nrow, ncol = {
            1: (1, 1),
            2: (1, 2),
            4: (2, 2),
        }[count]

        height = (
            7.2 if count < 4 else 11.
        ) if has_attractor_scan else (
            6.5 if count < 4 else 10.
        )

        fig = plt.figure(
            figsize=(12, height)
        )

        outer = fig.add_gridspec(
            nrow,
            ncol,
            left=.08,
            right=.73 if has_attractor_scan else .77,
            top=.94,
            bottom=.12 if has_attractor_scan else .10,
            hspace=.40,
            wspace=.30,
        )

        for index, data in enumerate(datasets):

            cell = outer[
                index//ncol,
                index%ncol,
            ]

            if has_attractor_scan:

                sub = cell.subgridspec(
                    3,
                    1,
                    height_ratios=[7, .8, .55],
                    hspace=.08,
                )

            else:

                sub = cell.subgridspec(
                    2,
                    1,
                    height_ratios=[7, .8],
                    hspace=.04,
                )

            upper = fig.add_subplot(sub[0])

            lower = fig.add_subplot(
                sub[1],
                sharex=upper,
            )

            # Original seagrass branches and periodic envelopes.
            draw_branches(
                upper,
                data,
                'S',
                True,
            )

            # Original strip showing communities without seagrass.
            draw_branches(
                lower,
                data,
                'S',
                True,
                True,
            )

            upper.set_title(
                f'H = {data["H"]:g} m'
            )

            upper.set_ylabel(
                'Seagrass S [mmol N m$^{-2}$]'
            )

            upper.tick_params(
                axis='x',
                labelbottom=False,
            )

            upper.set_ylim(bottom=0)

            lower.set_yticks(
                [0, 1, 2, 3],
                ['0', 'M', 'P', 'MP'],
            )

            lower.set_ylim(-.6, 3.6)

            lower.set_xlim(
                axis.low,
                axis.high,
            )

            if has_attractor_scan:

                lower.tick_params(
                    axis='x',
                    labelbottom=False,
                )

                attractors = fig.add_subplot(
                    sub[2],
                    sharex=upper,
                )

                draw_attractor_strip(
                    attractors,
                    data,
                    axis,
                )

                attractors.set_xlabel(
                    axis_label(axis)
                )

            else:

                lower.set_xlabel(
                    axis_label(axis)
                )

        # ------------------------------------------------------
        # Existing community and stability legend
        # ------------------------------------------------------

        used_communities = {
            path[0]['community']
            for data in datasets
            for path in data['paths']
            if path
        }

        handles = [
            Line2D(
                [], [],
                color=STATE_COLORS[community],
                lw=2,
                label=community,
            )
            for community in STATE_COLORS
            if community in used_communities
        ]

        handles += [
            Line2D(
                [], [],
                color='black',
                ls=STYLES[status],
                label=status.title(),
            )
            for status in (
                'stable',
                'unstable',
                'nonhyperbolic',
            )
        ]

        handles += periodic_handles(datasets)

        kinds = {
            event['kind']
            for data in datasets
            for event in data.get('events', [])
        }

        markers = {
            'fold': '^',
            'hopf': 'D',
            'invasion': 'o',
        }

        for kind in EVENT_STYLES:

            if kind not in kinds:
                continue

            handles.append(
                Line2D(
                    [], [],
                    marker=markers[kind],
                    ls='',
                    color=EVENT_STYLES[kind][2],
                    label=EVENT_STYLES[kind][3],
                )
            )

        if has_attractor_scan:

            handles += attractor_handles(
                datasets
            )

        fig.legend(
            handles=handles,
            loc='center left',
            bbox_to_anchor=(
                .75 if has_attractor_scan else .79,
                .53,
            ),
            frameon=False,
            fontsize=8 if has_attractor_scan else 9,
        )

        if has_attractor_scan:

            fig.text(
                .08,
                .025,
                'Attractor strip: sampled classifications. '
                'Shading between samples is a visual guide. '
                'An unsuccessful cycle search does not establish absence.',
                fontsize=8,
                va='bottom',
            )

        save(
            fig,
            folder,
            f'equilibrium_seagrass_states_{axis.name}',
        )


def polygon_patch(poly,axes,**kwargs):
    from shapely.geometry.polygon import orient
    poly=orient(poly,sign=1.)
    vertices,codes=[],[]
    for ring in (poly.exterior,*poly.interiors):
        xy=[(axes[0].physical(u),axes[1].physical(v)) for u,v in ring.coords]
        vertices.extend(xy);codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(xy)-2)+[MplPath.CLOSEPOLY])
    return PathPatch(MplPath(vertices,codes),**kwargs)


def draw_curves(ax,data):
    for curve in data['curves']:
        if len(curve)<2:continue
        style,width,color,_=EVENT_STYLES[curve[0]['kind']]
        ax.plot([r['x'] for r in curve],[r['y'] for r in curve],ls=style,lw=width,color=color,zorder=6)
        for endpoint in (curve[0],curve[-1]):
            if min(endpoint['u'],endpoint['v'],1-endpoint['u'],1-endpoint['v'])>1e-5:
                ax.plot(endpoint['x'],endpoint['y'],'x',color=color,ms=4,zorder=7)


def finish_axes(axes,data,controls):
    for ax,d in zip(axes.flat,data):
        ax.set_title(f'H = {d["H"]:g} m')
        ax.set_xlim(controls[0].low,controls[0].high);ax.set_ylim(controls[1].low,controls[1].high)
    for ax in axes[-1]:ax.set_xlabel(axis_label(controls[0]))
    for ax in axes[:,0]:ax.set_ylabel(axis_label(controls[1]))


def visible_region_labels(datasets,axes):
    """Legend entries for regions distinguishable at the figure's display scale.

    Tiny slivers remain in the scientific data and rendering, but do not suggest
    large visible states by adding misleading legend entries.
    """
    visible=set()
    for ax,data in zip(axes.flat,datasets):
        bounds=ax.get_window_extent()
        for region in data['regions']:
            poly=region['polygon'];x0,y0,x1,y1=poly.bounds
            if (poly.area*bounds.width*bounds.height>=16 and
                    (x1-x0)*bounds.width>=2 and (y1-y0)*bounds.height>=2):
                visible.add(region['label'])
    return visible


def plot_2d(datasets,controls,folder):
    labels=sorted({r['label'] for d in datasets for r in d['regions']})
    palette={label:BASE_COLORS.get(label,OTHER_COLORS[i%len(OTHER_COLORS)]) for i,label in enumerate(labels)}
    kinds={c[0]['kind'] for d in datasets for c in d['curves'] if len(c)>1}
    events=[Line2D([],[],ls=EVENT_STYLES[k][0],lw=EVENT_STYLES[k][1],color=EVENT_STYLES[k][2],
                   label=EVENT_STYLES[k][3]) for k in EVENT_STYLES if k in kinds]
    with plt.rc_context({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42}):
        for bw in (False,True):
            fig,axes=depth_axes(len(datasets))
            for ax,data in zip(axes.flat,datasets):
                for region in data['regions']:
                    label=region['label'];poly=region['polygon']
                    ax.add_patch(polygon_patch(poly,controls,facecolor=('white' if label!='?' else '#eeeeee') if bw else palette[label],edgecolor='none'))
                    if bw:
                        pt=poly.representative_point()
                        ax.text(controls[0].physical(pt.x),controls[1].physical(pt.y),label,
                                ha='center',va='center',fontsize=8,clip_on=True)
                draw_curves(ax,data)
                for cycle in data.get('oscillations',[]):
                    if cycle['status']=='verified_periodic':
                        stable=cycle['stability']=='stable'
                        ax.scatter(cycle['x'],cycle['y'],s=22,marker='o' if stable else '^',
                                   facecolor=STATE_COLORS['MSP'] if stable else 'none',edgecolor=STATE_COLORS['MSP'],linewidth=.7,zorder=9)
            finish_axes(axes,datasets,controls)
            display={'?':'Unresolved','∅':'No stable state found'}
            visible=visible_region_labels(datasets,axes)
            handles=events if bw else [Patch(facecolor=palette[l],label=display.get(l,l)) for l in labels if l in visible]+events
            cycle_labels=sorted({(r['stable_equilibria'],r['community']) for d in datasets
                                 for r in d.get('oscillations',[]) if r['status']=='verified_periodic' and r['stability']=='stable'})
            handles=handles+[Line2D([],[],ls='',marker='o',markeredgecolor='black',
                markerfacecolor=STATE_COLORS[c],label=(eq+' + ' if eq else '')+'○('+c+')')
                for eq,c in cycle_labels]
            if any(r['stability']!='stable' for d in datasets for r in d.get('oscillations',[])):
                handles.append(Line2D([],[],ls='',marker='^',markerfacecolor='none',color=STATE_COLORS['MSP'],label='Unstable / unresolved orbit'))
            if handles:fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.77,.52),frameon=False,fontsize=8)
            fig.text(.79,.12,'+ : alternative stable states\n○(MSP): stable periodic orbit\nMarkers: verified locations\n? / grey: equilibrium unresolved\n× : interior curve endpoint',fontsize=7,va='top')
            save(fig,folder,'stable_states_2D_BW_labels' if bw else 'stable_states_2D_color')
        for component in ('M','S','P_area','N_area'):
            fig,axes=depth_axes(len(datasets))
            all_values=[s['selected'][component] for d in datasets for r in d['regions'] if r['label']!='?'
                        for s in r['samples'] if s['selected']]
            norm=plt.Normalize(0,max(all_values,default=1) or 1)
            for ax,data in zip(axes.flat,datasets):
                for region in data['regions']:
                    samples=[s for s in region['samples'] if s['selected']]
                    if region['label']=='?' or len(samples)<3:
                        ax.add_patch(polygon_patch(region['polygon'],controls,facecolor='#eeeeee',edgecolor='none'));continue
                    uv=np.array([[s['u'],s['v']] for s in samples]);z=np.array([s['selected'][component] for s in samples])
                    lo_x,lo_y,hi_x,hi_y=region['polygon'].bounds
                    u,v=np.meshgrid(np.linspace(lo_x,hi_x,90),np.linspace(lo_y,hi_y,90))
                    try:interpolated=LinearNDInterpolator(uv,z)(u,v)
                    except QhullError:
                        # Narrow valid regions can have nearly collinear checks.
                        # Retain the checked values instead of aborting all plots.
                        interpolated=NearestNDInterpolator(uv,z)(u,v)
                    missing=~np.isfinite(interpolated)
                    interpolated[missing]=NearestNDInterpolator(uv,z)(u[missing],v[missing])
                    image=ax.imshow(interpolated,origin='lower',aspect='auto',norm=norm,cmap='viridis',
                        extent=[controls[0].physical(lo_x),controls[0].physical(hi_x),controls[1].physical(lo_y),controls[1].physical(hi_y)])
                    image.set_clip_path(polygon_patch(region['polygon'],controls,transform=ax.transData))
                draw_curves(ax,data)
            finish_axes(axes,datasets,controls)
            coloraxis=fig.add_axes([.81,.56,.025,.30])
            fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=coloraxis,
                         label=component.replace('_area','')+' [mmol N m$^{-2}$]')
            fig.text(.79,.46,'Selected stable state:\nmaximum seagrass\nGrey: unresolved/no\nstable equilibrium',fontsize=7,va='top')
            if events:fig.legend(handles=events,loc='upper left',bbox_to_anchor=(.77,.30),frameon=False,fontsize=8)
            save(fig,folder,f'biomass_2D_{component}')

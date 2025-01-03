import os
import warnings

import numpy as np
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astroquery.sdss import SDSS
from astroquery.sdss.core import NoResultsWarning

warnings.filterwarnings('ignore', category=NoResultsWarning)


def get_spec(plate:int, mjd:int, fiberid:int, model=True) -> dict:
    """
    Retrieves SDSS spectrum data for a given plate, mjd, and fiberid.

    Args:
        plate (int): Plate ID.
        mjd (int): Modified Julian Date.
        fiberid (int): Fiber ID.
        model (bool): Use model flux (True) or observed flux (False).

    Returns:
        dict: A dictionary with 'wavelength', 'flux', 'line_names', and 'line_waves'.
              Returns None if the spectrum cannot be retrieved.
    """
    
    hdul = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiberid)
    if hdul is None:
        print(f'No spectrum found for plate={plate}, mjd={mjd}, fiberID={fiberid}.')
        return None

    data = hdul[0]
    wav = 10 ** data[1].data.field('loglam')
    flux = data[1].data.field('model') if model else data[1].data.field('flux')

    line_area = data[3].data.field('LINEAREA')
    mask = (line_area != 0)

    line_names = data[3].data.field('LINENAME')[mask]
    line_waves_0 = data[3].data.field('LINEWAVE')[mask]
    line_z = data[3].data.field('LINEZ')[mask]
    line_waves = line_waves_0 * (1 + line_z)

    return {
        'wavelength': wav,
        'flux': flux,
        'line_names': np.char.strip(line_names),
        'line_waves': line_waves
    }


def make_spec_df(df, df_idx=None, model=True):
    """
    Builds a DataFrame containing spectral data for a given input DataFrame.

    Args:
        df (DataFrame): Input DataFrame containing 'plate', 'mjd', and 'fiberid'.
        df_idx (list, optional): Subset of indices to process.
        model (bool): Use model flux (True) or observed flux (False).

    Returns:
        DataFrame: A DataFrame with 'wavelength', 'flux', 'line_names', and 'line_waves'.
    """
    
    if df_idx is not None:
        df = df.loc[df_idx]
    
    specs = DataFrame(columns=['wavelength', 'flux', 'line_names', 'line_waves'])
    
    # search for plate, mjd, and fiberID columns
    plate_col = df.columns.str.lower() == 'plate'
    mjd_col = df.columns.str.lower() == 'mjd'
    fiberid_col = df.columns.str.lower() == 'fiberid'
    
    if plate_col.any() and mjd_col.any() and fiberid_col.any():
        plate_column = df.columns[plate_col][0]
        mjd_column = df.columns[mjd_col][0]
        fiberid_column = df.columns[fiberid_col][0]
    else:
        raise ValueError('Input DataFrame must contain plate, mjd, and fiberID columns.')
    
    for i in df.index.tolist():

        plate = df.loc[i, plate_column]
        mjd = df.loc[i, mjd_column]
        fiberid = df.loc[i, fiberid_column]
        
        spec = get_spec(plate=plate, mjd=mjd, fiberid=fiberid, model=model)
        if spec is not None:
            specs.loc[i] = spec['wavelength'], spec['flux'], spec['line_names'], spec['line_waves']
    
    return specs


def AB_to_flux(mag:float, err:int, wav:float) -> tuple:
    
    fv = 10 ** (-(mag + 2.41) / 2.5) / wav**2
    fv = fv * 1e17
    e_fv = np.log(10) * fv * err / 2.5
    
    return fv, e_fv


def plot_SED(ax, df, filters_dict, flux=True, errorbar=False):
    '''df: DataFrame with 1 row and columns of the 12 S-PLUS bands.'''
    
    if not filters_dict:
        filters_dict = {'u': '#CD00CD', 'J0378': '#610061', 'J0395': '#8000A1', 'J0410': '#7E00DB',
                        'J0430': '#3D00FF', 'g': '#00C0FF', 'J0515': '#1FFF00', 'r': '#FF6300',
                        'J0660': '#FF0000', 'i': '#D20000', 'J0861': '#610000', 'z': '#AA0000'}
    
    centers = {'u': 3536, 'J0378': 3770, 'J0395': 3940, 'J0410': 4094, 'J0430': 4292, 'g': 4751,
               'J0515': 5133, 'r': 6258, 'J0660': 6614, 'i': 7690, 'J0861': 8611, 'z': 8831}

    for name, color in filters_dict.items():
        
        if name in ['u', 'g', 'r', 'i', 'z']:
            marker = 'o'
        else:
            marker = '^'
        
        value = df[name+'_PStotal']
        wav = centers[name]
        try:
            error = df['e_'+name+'_PStotal']
        except KeyError:
            if errorbar:
                print('Errors not found.')
            error = 0
        
        if not flux and value == 99:
            continue
        
        if flux:
            value, error = AB_to_flux(value, error, wav)
            if error > 0.4*value:
                value, error = 0, 0
                mfc = 'none'
            else:
                mfc = color
        
        else:
            if error > 0.05*value:
                error = 0
                mfc = 'none'
            else:
                mfc = color
                
        if errorbar:
            ax.errorbar(wav, value, error, c=color, zorder=1, fmt=marker, ms=6, markerfacecolor=mfc)
        else:
            ax.scatter(wav, value, zorder=1, marker=marker, facecolor=mfc, edgecolor=color)


def plot_transm_curve(ax, filters_dict:dict, path_filters=os.path.join('filters')):
    ax2 = ax.twinx()
    
    for filter, color in filters_dict.items():
        filter_data = read_csv(os.path.join(path_filters, filter+'.dat'), sep=' ', names=['wav', filter])
        filter_data = filter_data.query(f'{filter} > 0.0001')
        ax2.fill_between(filter_data['wav'], filter_data[filter], alpha=0.08, color=color)
        
    ax2.set_ylim(0, 0.775)
    return ax2


def plot_lines(ax, spec:dict, lines:list|dict|None, grid:bool):
    
    if isinstance(lines, list):
        lines_to_plot = lines
    elif isinstance(lines, dict):
        lines_to_plot = list(lines.keys())
    elif lines is None:
        lines_to_plot = spec['line_names'].tolist()
    else:
        print('Invalid input type for lines.')
        return None
    
    indices = np.where(np.isin(spec['line_names'], lines_to_plot))
    
    if len(indices[0]) > 0:
        line_waves = spec['line_waves'][indices[0]]
        line_names = spec['line_names'][indices[0]]
        if len(line_waves) > 1:
            color_index = (line_waves - line_waves.min()) / (line_waves.max() - line_waves.min())
        else:
            color_index = [0.1]
        
        colors = []   
        for i in range(len(line_waves)):
            
            if isinstance(lines, list) or lines is None:
                color = plt.cm.get_cmap('tab20')(color_index[i])
            else:
                color = lines[line_names[i]]
                        
            if not grid:
                ax.axvline(line_waves[i], ls='--', label=line_names[i], color=color, zorder=2, lw=2)
            else:
                ax.axvline(line_waves[i], ls='--', color=color, zorder=2, lw=2)
                colors.append(color)
        
        if not grid:
            ax.legend(ncol=5)
            handles, _ = ax.get_legend_handles_labels()
            for handle in handles:
                handle.set_linewidth(2)        
        else:
            return list(line_names), colors


def plot_spec(ax, df, spec:dict, lines=[], SED=False, transm_curve={}, legend_cols=None, grid=False):
    
    if spec is not None:
        ax.plot(spec['wavelength'], spec['flux'], color='k', lw=1, zorder=1)
        if lines or lines is None:
            plot_lines(ax, spec, lines, grid)
    
    if legend_cols is None:
        legend_cols = {'Z': '$z_{spec}$', 'r_PStotal': '$r$'}
    
    legend_items = []
    for col, display_name in legend_cols.items():
        try:
            try:
                value = round(df[col], 2)
            except TypeError:
                value = df[col]
            legend_items.append(f'{display_name} = {value}')
        except KeyError:
            continue
    
    if not grid:
        ax.set_title(', '.join(legend_items), size=16)
    else:
        ax.legend([Line2D([0], [0])], ['\n'.join(legend_items)], handlelength=0, handletextpad=0, fontsize=16)
    
    ax.grid(axis='x', alpha=0.5)
    
    if SED:
        plot_SED(ax, df, transm_curve, flux=True)
    if transm_curve:
        ax2 = plot_transm_curve(ax, transm_curve)
        ax.set_zorder(ax2.get_zorder()+1)
        ax.set_frame_on(False)
        return ax2
    
    return ax


def grid_specs(df, df_idx=None, specs=None, lines={}, SED=False, transm_curve={}, model=True, legend_cols=None):
    
    if df_idx is not None:
        df = df.loc[df_idx]
    index = df.index
    
    ncols = 2
    nrows = (len(df) // 2) + (len(df) % 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3*nrows), sharex=True)
    axes = axes.flatten()
    
    if specs is None:
        specs = make_spec_df(
            df=df,
            df_idx=df_idx,
            model=model
            )
    
    lines_dict = {}
        
    for i in range(len(df)):
        idx = index[i]
        ax = axes[i]
        
        if i >= (nrows-1)*ncols:
            ax.set_xlabel('Wavelength [Å]', size=16)
        
        try:
            spec = specs.loc[idx].to_dict()
            if lines or lines is None:
                labels_colors = plot_lines(ax, spec, lines, grid=True)
                if labels_colors is not None:
                    labels, colors = labels_colors
                    lines_dict.update({labels[i]: colors[i] for i in range(len(labels))})
        except KeyError:
            spec = None
        
        ax = plot_spec(
            ax=ax,
            df=df.loc[idx],
            spec=spec,
            lines=False,
            SED=SED,
            transm_curve=transm_curve,
            legend_cols=legend_cols,
            grid=True
            )
        
        if transm_curve:
            if i % ncols == 0:
                ax.set_yticklabels([])
                ax.set_yticks([])
    
    if len(lines_dict) > 0:
        handles = [Line2D([0], [0], ls='--', c=lines_dict[line]) for line in lines_dict]
        fig.legend(handles=handles, labels=lines_dict.keys(), loc='upper center',
                        bbox_to_anchor=(0.5, 1.14), ncol=8, fontsize=16)
        for handle in handles:
            handle.set_linewidth(2)

    fig.text(-0.01, 0.5, 'Flux [$10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', va='center', rotation='vertical', size=16)
    if transm_curve:
        fig.text(1.01, 0.5, 'Transmittance [%]', va='center', rotation='vertical', size=16)
    fig.tight_layout()
    plt.show()

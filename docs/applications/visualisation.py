'''
Tools for converting plots into formats viewable in the sphinx HTML documentation

'''

import numpy as np
# interactive visualisation  inside the notebook with nglview
from nglview import show_ase, ASEStructure
import matplotlib.pyplot as plt
from IPython.display import HTML
def show_HTML(anim):
    '''
    Convert matplotlib.animation.FuncAnimation 
    (e.g. from GammaSurface.show()) to HTML viewable object.

    anim: matplotlib.animation.FuncAnimation object
        Animation to convert to HTML
    '''
    html = anim.to_jshtml()
    # center the animation according to the width of the page
    # does not affect the size of the figure
    output_html = f'''
        <div style="display: flex; justify-content: center;">
        {html}
        </div>
        '''
    plt.close(fig=plt.gcf())
    return HTML(output_html)


# Get the results of Ovito Common Neighbor Analysis 
# https://www.ovito.org/docs/current/reference/pipelines/modifiers/common_neighbor_analysis.html
# and Identify Diamond modifier
# https://www.ovito.org/docs/current/reference/pipelines/modifiers/identify_diamond.html
# for better visualisation of the dislocation core
# it will be identified as "other" structure type
def get_structure_types(structure, diamond_structure=False):
    """Get the results of Common Neighbor Analysis and 
        Identify Diamond modifiers from Ovito
    Args:
        structure (ase.atoms): input structure
    Returns:
        atom_labels (array of ints): per atom labels of the structure types
        structure_names (list of strings): names of the structure types
        colors (list of strings): colors of the structure types in hex format
    """
    from ovito.io.ase import ase_to_ovito
    from ovito.modifiers import CommonNeighborAnalysisModifier, IdentifyDiamondModifier
    from ovito.pipeline import StaticSource, Pipeline
    ovito_structure = structure.copy()
    if "fix_mask" in ovito_structure.arrays:
        del ovito_structure.arrays["fix_mask"]
    
    if diamond_structure:
        modifier = IdentifyDiamondModifier()
    else:
        modifier = CommonNeighborAnalysisModifier() 
    
    data = ase_to_ovito(ovito_structure)
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()

    atom_labels = data.particles['Structure Type'].array

    structure_names = [structure.name for structure in modifier.structures]
    colors = [structure.color for structure in modifier.structures]
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in colors] 

    return atom_labels, structure_names, hex_colors

# custom tooltip for nglview to avoid showing molecule and residue names
# that are not relevant for our type of structures
tooltip_js = """
                this.stage.mouseControls.add('hoverPick', (stage, pickingProxy) => {
                    let tooltip = this.stage.tooltip;
                    if(pickingProxy && pickingProxy.atom && !pickingProxy.bond){
                        let atom = pickingProxy.atom;
                        if (atom.structure.name.length > 0){
                            tooltip.innerText = atom.atomname + " atom: " + atom.structure.name;
                        } else {
                            tooltip.innerText = atom.atomname + " atom";
                        }
                    } else if (pickingProxy && pickingProxy.bond){
                        let bond = pickingProxy.bond;
                        if (bond.structure.name.length > 0){
                        tooltip.innerText = bond.atom1.atomname + "-" + bond.atom2.atomname + " bond: " + bond.structure.name;
                        } else {
                            tooltip.innerText = bond.atom1.atomname + "-" + bond.atom2.atomname + " bond";
                        }
                    } else if (pickingProxy && pickingProxy.unitcell){
                        tooltip.innerText = "Unit cell";
                    }
                });
                """

def add_dislocation(view, system, name, color=[0, 1, 0], x_shift=0.0):
    '''Add dislocation line to the view as a cylinder and two cones.
    The cylinder is hollow by default so the second cylinder is needed to close it.
    In case partial distance is provided, two dislocation lines are added and are shifter accordingly.
    '''
    
    center = system.positions.mean(axis=0)
    view.shape.add_cylinder((center[0] + x_shift, center[1], -2.0), 
                            (center[0] + x_shift, center[1], system.cell[2][2] - 0.5),
                            color,
                            [0.3],
                            name)
    
    view.shape.add_cone((center[0] + x_shift, center[1], -2.0), 
                        (center[0] + x_shift, center[1], 0.5),
                        color,
                        [0.3],
                        name)
       
    view.shape.add_cone((center[0] + x_shift, center[1], system.cell[2][2] - 0.5), 
                        (center[0] + x_shift, center[1], system.cell[2][2] + 1.0),
                        color,
                        [0.55],
                        name)
    

def show_dislocation(system, scale=0.5, CNA_color=True, diamond_structure=False, add_bonds=False,
                     d_name='', d_color=[0, 1, 0], partial_distance=None):

    atom_labels, structure_names, colors = get_structure_types(system, 
                                                               diamond_structure=diamond_structure)

    view = show_ase(system)
    view.hide([0])
    
    if add_bonds: # add bonds between all atoms to have bonds between structures
        component = view.add_component(ASEStructure(system), default_representation=False, name='between structures')
        component.add_ball_and_stick(cylinderOnly=True, radiusType='covalent', radiusScale=scale, aspectRatio=0.1)
    
    for structure_type in np.unique(atom_labels):
        # every structure type is a different component
        mask = atom_labels == structure_type
        component = view.add_component(ASEStructure(system[mask]), 
                                       default_representation=False, name=str(structure_names[structure_type]))
        if CNA_color:
            if add_bonds:
                component.add_ball_and_stick(color=colors[structure_type], radiusType='covalent', radiusScale=scale)
            else:
                component.add_spacefill(color=colors[structure_type], radiusType='covalent', radiusScale=scale)
        else:
            if add_bonds:
                component.add_ball_and_stick(radiusType='covalent', radiusScale=scale)
            else:
                component.add_spacefill(radiusType='covalent', radiusScale=scale)
                    
    component.add_unitcell()

    if partial_distance is None:
        add_dislocation(view, system, d_name + ' dislocation line', d_color)
    else:
        if type(d_name) == list:
            add_dislocation(view, system, d_name[0] + ' dislocation line', d_color, x_shift= -partial_distance / 2.0)
            add_dislocation(view, system, d_name[1] + ' dislocation line', d_color, x_shift= partial_distance / 2.0)
        else:
            add_dislocation(view, system, d_name + ' dislocation line', d_color, x_shift= -partial_distance / 2.0)
            add_dislocation(view, system, d_name + ' dislocation line', d_color, x_shift= partial_distance / 2.0)

    view.camera = 'orthographic'
    view.parameters = {"clipDist": 0}

    #view._remote_call("setSize", target="Widget", args=["400px", "300px"])
    view.layout.width = '100%'
    view.layout.height = '300px'
    view.center()

    view._js(tooltip_js)
    return view


def interactive_view(system, scale=0.5, name=""):
    view = show_ase(system)
    view._remove_representation()
    component = view.add_component(ASEStructure(system), 
                                   default_representation=False, name=name)
    component.add_spacefill(radiusType='covalent', radiusScale=scale)
    view.add_unitcell()
    view.update_spacefill(radiusType='covalent',
                          radiusScale=scale)

    view.camera = 'orthographic'
    view.parameters = {"clipDist": 0}

    view.center()
    #view._remote_call("setSize", target="Widget", args=["300px", "300px"])
    view.layout.width = '100%'
    view.layout.height = '300px'
    view._js(tooltip_js)
    return view
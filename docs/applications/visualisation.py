'''
Tools for converting plots into formats viewable in the sphinx HTML documentation

'''

import numpy as np
# interactive visualisation  inside the notebook with nglview
import matplotlib.pyplot as plt
from IPython.display import HTML
from nglview import show_ase, ASEStructure

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

def interactive_view(system, scale=0.5, name=""):
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
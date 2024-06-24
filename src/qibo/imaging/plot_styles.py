def _get_style(style_name):

        if 'garnacha' in style_name:
            new_values = dict()
            new_values["facecolor"]="#5e2129"
            new_values["edgecolor"]="#FFFFFF"
            new_values["linecolor"]="#FFFFFF"
            new_values["textcolor"]="#FFFFFF"
            new_values["fillcolor"]="#FFFFFF"
            new_values["gatecolor"]="#5e2129"
            new_values["controlcolor"]="#FFFFFF"
            return new_values

        if 'fardelejo' in style_name:
            new_values = dict()
            new_values["facecolor"]="#e17a02"
            new_values["edgecolor"]="#fef1e2"
            new_values["linecolor"]="#fef1e2"
            new_values["textcolor"]="#FFFFFF"
            new_values["fillcolor"]="#fef1e2"
            new_values["gatecolor"]="#8b4513"
            new_values["controlcolor"]="#fef1e2"
            return new_values

        if 'quantumspain' in style_name:
            new_values = dict()
            new_values["facecolor"]="#EDEDF4"
            new_values["edgecolor"]="#092D4E"
            new_values["linecolor"]="#092D4E"
            new_values["textcolor"]="#8561C3"
            new_values["fillcolor"]="#092D4E"
            new_values["gatecolor"]="#53E7CA"
            new_values["controlcolor"]="#092D4E"
            return new_values

        if 'color-blind' in style_name:
            new_values = dict()
            new_values["facecolor"]="#d55e00"
            new_values["edgecolor"]="#f0e442"
            new_values["linecolor"]="#f0e442"
            new_values["textcolor"]="#f0e442"
            new_values["fillcolor"]="#cc79a7"
            new_values["gatecolor"]="#d55e00"
            new_values["controlcolor"]="#f0e442"
            return new_values

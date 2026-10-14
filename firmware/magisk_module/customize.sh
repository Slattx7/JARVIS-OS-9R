SKIPUNZIP=0

ui_print " "
ui_print "  ██████╗ ███╗   ███╗███╗   ██╗██╗"
ui_print " ██╔═══██╗████╗ ████║████╗  ██║██║"
ui_print " ██║   ██║██╔████╔██║██╔██╗ ██║██║"
ui_print " ██║   ██║██║╚██╔╝██║██║╚██╗██║██║"
ui_print " ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║"
ui_print "  ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝"
ui_print " "
ui_print "===================================="
ui_print "       PROJECT OMNI EDR V1.0       "
ui_print "===================================="
ui_print " "

ui_print "- Checking device architecture..."
if [ "$ARCH" != "arm64" ]; then
  ui_print "! Project OMNI requires an arm64 device (found $ARCH)!"
  abort "! Installation aborted."
fi

ui_print "- Checking Android API level..."
if [ "$API" -lt 29 ]; then
  ui_print "! Project OMNI requires Android 10+ for eBPF support!"
  abort "! Installation aborted."
fi

ui_print "- Extracting OMNI Binaries..."
# Since SKIPUNZIP=0, Magisk automatically extracts everything to $MODPATH.

ui_print "- Setting executable permissions..."
set_perm_recursive $MODPATH/root/ 0 0 0755 0755
set_perm $MODPATH/service.sh 0 0 0755

ui_print "- Installation Complete!"
ui_print "- Reboot to activate Project OMNI."

/*
 * warden behavioral hook agent.
 *
 * Injected into a target app's process via Frida. Every hook is wrapped in
 * its own try/catch so a single missing API (OS version differences,
 * OEM/AOSP fork quirks) can't take down the rest of the agent. Each hook
 * calls through to the original implementation after emitting an event, so
 * the target app's real behavior is never altered - this is passive
 * observation only.
 *
 * Emitted events are plain JSON sent over Frida's message channel; the
 * Python side (collector.py) is the only thing that interprets them.
 *
 * The pure classification helpers below live at top level (outside the
 * Java bridge) on purpose: that lets tests run them under plain Node
 * without a device.
 */

// ---- pure helpers (no Frida/Java dependencies) -------------------------

var ROOT_BINARY_NAMES = ['su', 'busybox'];
var PATH_SUBSTRING_KINDS = [
    ['magisk', 'magisk'],
    ['frida', 'frida'],
    ['supersu', 'superuser'],
    ['superuser', 'superuser'],
    ['lsposed', 'xposed'],
    ['edxposed', 'xposed'],
    ['xposed', 'xposed']
];

// Classifies a filesystem path an app is checking. Returns a category
// string if it looks like a root/Frida/hooking-framework artifact probe,
// otherwise null (so ordinary File.exists() traffic is never logged).
function classifyPath(path) {
    if (!path) return null;
    var lower = String(path).toLowerCase();
    var base = lower.split('/').pop();
    if (ROOT_BINARY_NAMES.indexOf(base) !== -1) return 'root_binary';
    for (var i = 0; i < PATH_SUBSTRING_KINDS.length; i++) {
        if (lower.indexOf(PATH_SUBSTRING_KINDS[i][0]) !== -1) return PATH_SUBSTRING_KINDS[i][1];
    }
    return null;
}

// Classifies a shell command passed to Runtime.exec. Same contract as
// classifyPath: category string for probe-looking commands, else null.
function classifyCommand(cmd) {
    if (!cmd) return null;
    var lower = String(cmd).toLowerCase();
    if (/(^|[\s\/])su(\s|$)/.test(lower)) return 'root_binary';
    if (lower.indexOf('magisk') !== -1) return 'magisk';
    if (lower.indexOf('frida') !== -1) return 'frida';
    if (lower.indexOf('busybox') !== -1) return 'root_binary';
    if (/getprop\s+ro\.(debuggable|secure|build\.tags|build\.fingerprint)/.test(lower)) return 'system_property';
    if (/^ps(\s|$)/.test(lower) || /\|\s*grep/.test(lower)) return 'process_listing';
    return null;
}

// Flattens the first argument of Runtime.exec (String or String[]) to one string.
function stringifyCommand(arg) {
    try {
        if (arg === null || arg === undefined) return '';
        if (typeof arg === 'string') return arg;
        if (arg.length !== undefined) {
            var parts = [];
            for (var i = 0; i < arg.length; i++) parts.push(String(arg[i]));
            return parts.join(' ');
        }
        return arg.toString();
    } catch (e) {
        return '';
    }
}

var ADB_SETTING_NAMES = ['adb_enabled', 'development_settings_enabled'];

// ---- Frida-side hooks --------------------------------------------------

function main() {
    function emit(eventType, details) {
        send({
            event_type: eventType,
            timestamp: new Date().toISOString(),
            details: details || {}
        });
    }

    function hookError(hookName, err) {
        send({
            event_type: 'HOOK_ERROR',
            timestamp: new Date().toISOString(),
            details: { hook: hookName, error: err.toString() }
        });
    }

    // --- Runtime permission usage (AppOpsManager is the actual gate Android
    // enforces on dangerous-permission use, not just the manifest declaration) ---
    try {
        var AppOpsManager = Java.use('android.app.AppOpsManager');
        ['noteOp', 'noteOpNoThrow'].forEach(function (methodName) {
            if (!AppOpsManager[methodName]) return;
            AppOpsManager[methodName].overloads.forEach(function (overload) {
                overload.implementation = function () {
                    var args = Array.prototype.slice.call(arguments).map(function (a) {
                        try { return a === null || a === undefined ? null : a.toString(); }
                        catch (e) { return String(a); }
                    });
                    emit('APP_OP', { method: methodName, args: args });
                    return this[methodName].apply(this, arguments);
                };
            });
        });
    } catch (e) {
        hookError('AppOpsManager', e);
    }

    // --- Network: destination host/port at socket-address construction time.
    // Catches this ahead of the TLS handshake, so it works for HTTPS traffic
    // too without needing to touch the encrypted payload. ---
    try {
        var InetSocketAddress = Java.use('java.net.InetSocketAddress');
        InetSocketAddress.$init.overload('java.lang.String', 'int').implementation = function (host, port) {
            emit('NETWORK_CONNECT', { host: host, port: port });
            return this.$init(host, port);
        };
    } catch (e) {
        hookError('InetSocketAddress', e);
    }

    // --- Sensors: which sensor type gets registered, regardless of which
    // registerListener overload the app used. ---
    try {
        var SensorManager = Java.use('android.hardware.SensorManager');
        SensorManager.registerListener.overloads.forEach(function (overload) {
            overload.implementation = function () {
                var args = Array.prototype.slice.call(arguments);
                var sensorType = -1;
                for (var i = 0; i < args.length; i++) {
                    if (args[i] && typeof args[i].getType === 'function') {
                        try { sensorType = args[i].getType(); } catch (e) { /* ignore */ }
                        break;
                    }
                }
                emit('SENSOR_REGISTER', { sensor_type: sensorType });
                return this.registerListener.apply(this, arguments);
            };
        });
    } catch (e) {
        hookError('SensorManager', e);
    }

    // --- Location updates: capture the provider string if the overload
    // used one, plus the raw argument count so unusual overloads still
    // show up as *something*. ---
    try {
        var LocationManager = Java.use('android.location.LocationManager');
        LocationManager.requestLocationUpdates.overloads.forEach(function (overload) {
            overload.implementation = function () {
                var args = Array.prototype.slice.call(arguments);
                var provider = null;
                for (var i = 0; i < args.length; i++) {
                    if (typeof args[i] === 'string') { provider = args[i]; break; }
                }
                emit('LOCATION_REQUEST', { provider: provider, arg_count: args.length });
                return this.requestLocationUpdates.apply(this, arguments);
            };
        });
    } catch (e) {
        hookError('LocationManager', e);
    }

    // --- Environment probes: the app checking whether it is being watched
    // (root, Frida, hooking frameworks, debugger, ADB). The probing itself
    // is the signal - it needs no behavior change on the app's part, and
    // only probe-looking paths/commands are ever emitted. ---
    ['exists', 'canExecute'].forEach(function (methodName) {
        try {
            var File = Java.use('java.io.File');
            File[methodName].implementation = function () {
                var category = classifyPath(this.getPath());
                if (category) {
                    emit('ENV_PROBE', { kind: 'file_' + methodName, target: String(this.getPath()), category: category });
                }
                return this[methodName]();
            };
        } catch (e) {
            hookError('File.' + methodName, e);
        }
    });

    try {
        var Runtime = Java.use('java.lang.Runtime');
        Runtime.exec.overloads.forEach(function (overload) {
            overload.implementation = function () {
                var cmd = stringifyCommand(arguments[0]);
                var category = classifyCommand(cmd);
                if (category) {
                    emit('ENV_PROBE', { kind: 'exec', target: cmd.substring(0, 200), category: category });
                }
                return this.exec.apply(this, arguments);
            };
        });
    } catch (e) {
        hookError('Runtime.exec', e);
    }

    try {
        var Debug = Java.use('android.os.Debug');
        Debug.isDebuggerConnected.implementation = function () {
            emit('ENV_PROBE', { kind: 'debugger_check', target: 'isDebuggerConnected', category: 'debugger' });
            return this.isDebuggerConnected();
        };
    } catch (e) {
        hookError('Debug.isDebuggerConnected', e);
    }

    try {
        var SettingsGlobal = Java.use('android.provider.Settings$Global');
        SettingsGlobal.getInt.overloads.forEach(function (overload) {
            overload.implementation = function () {
                var name = arguments.length > 1 ? String(arguments[1]) : '';
                if (ADB_SETTING_NAMES.indexOf(name) !== -1) {
                    emit('ENV_PROBE', { kind: 'settings_check', target: name, category: 'adb' });
                }
                return this.getInt.apply(this, arguments);
            };
        });
    } catch (e) {
        hookError('Settings.Global.getInt', e);
    }

    emit('AGENT_READY', {});
}

if (typeof Java !== 'undefined') {
    Java.perform(main);
}

// Test seam: under plain Node (no Java global), expose the pure helpers.
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { classifyPath: classifyPath, classifyCommand: classifyCommand, stringifyCommand: stringifyCommand };
}

import json, os, unittest
from helpers import *
from jarvis_actions.backends.adb import Adb, AdbError
from jarvis_actions.backends.root import RootShell
from jarvis_actions.backends.frida_be import Frida, FridaScripts, ScriptError, build_manifest
from jarvis_actions.backends.local import FsSandbox, FsError
from jarvis_actions.runner import FakeRunner, RunResult

UI = '<?xml version="1.0"?><hierarchy rotation="0"><node text="Hello" content-desc="" package="x"/><node text="" content-desc="Send money" package="x"/></hierarchy>UI hierarchy dumped to: /dev/tty'


class AdbT(unittest.TestCase):
    def test_argv_quoting(self):
        r = FakeRunner(); a = Adb(r, "SER")
        a.tap(10, 20); a.text("hi there; rm -rf /"); a.open_app("com.a.b"); a.compose_sms("+911234567", "it's me")
        self.assertEqual(r.calls[0], ["adb", "-s", "SER", "shell", "input tap 10 20"])
        self.assertEqual(r.calls[1][-1], "input text 'hi%sthere;%srm%s-rf%s/'")      # spaces->%s, quoted for device sh
        self.assertIn("monkey -p com.a.b", r.calls[2][-1])
        self.assertIn("'it'\"'\"'s me'", r.calls[3][-1])

    def test_validation(self):
        a = Adb(FakeRunner())
        for f in (lambda: a.open_app("x; id"), lambda: a.open_url("file:///etc/passwd"), lambda: a.keyevent("66; id"),
                  lambda: a.dial("12"), lambda: a.text("héllo"), lambda: a.uninstall("..")):
            with self.assertRaises(ValueError): f()

    def test_failure_raises(self):
        a = Adb(FakeRunner(lambda argv: RunResult(argv, 1, b"", b"device offline")))
        with self.assertRaises(AdbError): a.tap(1, 1)

    def test_screen_parse(self):
        a = Adb(FakeRunner(lambda argv: UI if "uiautomator" in argv else ""))
        self.assertEqual(a.screen_nodes(), ["Hello", "Send money"])

    def test_shizuku_wrap(self):
        r = FakeRunner(); a = Adb(r, "S", shell_prefix=["sh", "/data/local/tmp/rish"], env={"RISH_APPLICATION_ID": "com.android.shell"})
        a.tap(1, 2)
        self.assertEqual(r.calls[0][-1], "RISH_APPLICATION_ID=com.android.shell sh /data/local/tmp/rish -c 'input tap 1 2'")

    def test_root_shell(self):
        r = FakeRunner(); rs = RootShell(Adb(r, "S")); rs.run("id; whoami")
        self.assertEqual(r.calls[0][-1], "su -c 'id; whoami'")
        r2 = FakeRunner(); RootShell(Adb(r2, "S"), "0").run("id")
        self.assertEqual(r2.calls[0][-1], "su 0 sh -c id")
        with self.assertRaises(ValueError): RootShell(Adb(r), "bogus")

    def test_composite_actions(self):
        r = FakeRunner(); a = Adb(r, "SER")
        a.youtube_play("kendrick lamar")
        self.assertIn("MEDIA_PLAY_FROM_SEARCH -e query 'kendrick lamar' com.google.android.youtube", r.calls[-1][-1])
        a.spotify_play("the weeknd")
        self.assertIn("MEDIA_PLAY_FROM_SEARCH -e query 'the weeknd' com.spotify.music", r.calls[-1][-1])
        with self.assertRaises(ValueError): a.instagram_dm("bad user with spaces!", "hello")

    def test_god_mode_primitives(self):
        r = FakeRunner(); a = Adb(r, "SER")
        self.assertEqual(a.device_home(), {"status": "home"})
        self.assertIn("input keyevent KEYCODE_HOME", r.calls[-1][-1])

        self.assertEqual(a.device_back(), {"status": "back"})
        self.assertIn("input keyevent KEYCODE_BACK", r.calls[-1][-1])

        self.assertEqual(a.wake_screen(), {"status": "awake"})
        self.assertIn("input keyevent KEYCODE_WAKEUP", r.calls[-1][-1])

        self.assertEqual(a.close_app("com.brave.browser"), {"status": "stopped", "package": "com.brave.browser"})
        self.assertIn("am force-stop com.brave.browser", r.calls[-1][-1])
        with self.assertRaises(ValueError): a.close_app("invalid;package")

        # Battery test
        bat_out = "Current Battery Service state:\n  level: 85\n  temperature: 325\n  status: 2\n  health: 2\n"
        a_bat = Adb(FakeRunner(lambda argv: bat_out if "dumpsys battery" in argv[-1] else ""))
        bat = a_bat.battery_status()
        self.assertEqual(bat["level"], 85)
        self.assertEqual(bat["temperature"], 325)
        self.assertEqual(bat["temp_c"], 32.5)

        # Screen elements & click label test
        ui_xml = ('<?xml version="1.0"?><hierarchy rotation="0">'
                  '<node text="Search" bounds="[100,200][300,400]" class="android.widget.Button"/>'
                  '<node text="" content-desc="Profile" bounds="[400,500][600,700]" class="android.widget.ImageView"/>'
                  '</hierarchy>UI hierarchy dumped to: /dev/tty')
        a_ui = Adb(FakeRunner(lambda argv: ui_xml if "uiautomator" in argv else ""))
        elems = a_ui.screen_elements()
        self.assertEqual(len(elems), 2)
        self.assertEqual(elems[0]["label"], "Search")
        self.assertEqual(elems[0]["pos"], [200, 300])

        click_res = a_ui.click_label("Search")
        self.assertEqual(click_res["status"], "clicked")
        self.assertEqual(click_res["x"], 200)
        self.assertEqual(click_res["y"], 300)

        # List user apps test
        pm_out = "package:com.brave.browser\npackage:com.instagram.android\n"
        a_pm = Adb(FakeRunner(lambda argv: pm_out if "pm list packages" in argv[-1] else ""))
        self.assertEqual(a_pm.list_user_apps(), ["com.brave.browser", "com.instagram.android"])


class FridaT(unittest.TestCase):
    def setUp(self):
        import tempfile; self.d = tempfile.mkdtemp()
        open(os.path.join(self.d, "hello.js"), "w").write("console.log(Process.id)")
        json.dump(build_manifest(self.d), open(os.path.join(self.d, "manifest.json"), "w"))

    def test_hash_pin(self):
        s = FridaScripts(self.d); p = s.path("hello"); self.assertTrue(p.endswith("hello.js"))
        open(p, "a").write("//tampered")
        with self.assertRaises(ScriptError): s.path("hello")
        with self.assertRaises(ScriptError): s.path("missing")

    def test_manifest_escape(self):
        s = FridaScripts(self.d); s.manifest["evil"] = {"file": "../../etc/passwd", "sha256": "0"}
        with self.assertRaises(ScriptError): s.path("evil")

    def test_argv(self):
        f = Frida(FakeRunner(), FridaScripts(self.d), "SER")
        argv = f.argv("com.a.b", "hello", 12)
        self.assertEqual(argv[:5], ["frida", "-D", "SER", "-f", "com.a.b"]); self.assertIn("-q", argv)
        with self.assertRaises(ValueError): f.argv("com.a.b;id", "hello", 5)
        self.assertEqual(Frida(FakeRunner(), FridaScripts(self.d), None, mode="attach").argv("a.b", "hello", 5)[1:4], ["-U", "-n", "a.b"])


class FsT(unittest.TestCase):
    def setUp(self):
        import tempfile; t = tempfile.mkdtemp(); self.t = t
        os.makedirs(f"{t}/data"); open(f"{t}/data/a.txt", "w").write("hi"); open(f"{t}/secret.txt", "w").write("s")
        self.fs = FsSandbox([f"{t}/data"], f"{t}/scratch")

    def test_confinement(self):
        self.assertEqual(self.fs.list(f"{self.t}/data"), ["a.txt"]); self.assertEqual(self.fs.read(f"{self.t}/data/a.txt"), "hi")
        for p in (f"{self.t}/secret.txt", f"{self.t}/data/../secret.txt", "/etc/passwd"):
            with self.assertRaises(FsError): self.fs.read(p)

    def test_symlink_escape(self):
        os.symlink(f"{self.t}/secret.txt", f"{self.t}/data/link")
        with self.assertRaises(FsError): self.fs.read(f"{self.t}/data/link")

    def test_scratch_write_undo_and_names(self):
        u = self.fs.write_scratch("n.txt", "v1"); u2 = self.fs.write_scratch("n.txt", "v2")
        self.assertEqual(self.fs.read(f"{self.t}/scratch/n.txt"), "v2"); u2(); self.assertEqual(self.fs.read(f"{self.t}/scratch/n.txt"), "v1")
        u(); self.assertFalse(os.path.exists(f"{self.t}/scratch/n.txt"))
        for bad in ("../x", "a/b", ".hid", ""):
            with self.assertRaises(FsError): self.fs.write_scratch(bad, "x")

    def test_delete_trash_undo(self):
        self.fs.write_scratch("d.txt", "x"); p = f"{self.t}/scratch/d.txt"
        undo = self.fs.delete(p); self.assertFalse(os.path.exists(p)); undo(); self.assertTrue(os.path.exists(p))
        with self.assertRaises(FsError): self.fs.delete(f"{self.t}/data/a.txt")      # outside delete roots
        with self.assertRaises(FsError): self.fs.delete(f"{self.t}/scratch")         # a root
        self.assertTrue(self.fs.in_scratch(p)); self.assertFalse(self.fs.in_scratch("/etc/passwd"))

if __name__ == "__main__": unittest.main()

'use strict';
// EXPERIMENTAL -- real-time notification sensor. Hooks NotificationManagerService.enqueueNotificationInternal
// inside system_server. The signature differs across Android versions and ROMs, so this script:
//   * enumerates candidate overloads first (ENUM_ONLY = true: hooks NOTHING, only reports what it found), and
//   * hooks only overloads shaped like (String pkg, ..., String tag, int id, Notification n, ...).
// A bad hook in system_server can soft-reboot the phone. Run with --enum first, read the candidates, then --hook.
// The bridge substitutes __ENUM_ONLY__ with true/false.
var ENUM_ONLY = __ENUM_ONLY__;
var CLS = 'com.android.server.notification.NotificationManagerService';

function str(v) { try { return (v === null || v === undefined) ? '' : v.toString(); } catch (e) { return ''; } }

Java.perform(function () {
  var NMS;
  try { NMS = Java.use(CLS); } catch (e) { send({kind: 'error', msg: 'class not found: ' + e}); return; }
  var cands = NMS.class.getDeclaredMethods().map(function (m) { return m.toString(); })
    .filter(function (s) { return s.indexOf('enqueueNotificationInternal') !== -1; });
  send({kind: 'enum', candidates: cands});
  if (ENUM_ONLY || !cands.length) return;

  var hooked = 0;
  NMS.enqueueNotificationInternal.overloads.forEach(function (ov) {
    var types = ov.argumentTypes.map(function (t) { return t.className; });
    var ni = types.indexOf('android.app.Notification');
    if (ni < 2 || types[0] !== 'java.lang.String') return;
    ov.implementation = function () {
      try {
        var n = arguments[ni];
        var ex = n.extras.value;
        send({kind: 'notification', pkg: str(arguments[0]), tag: str(arguments[ni - 2]), id: Number(arguments[ni - 1]),
              flags: n.flags.value, category: str(n.category.value), channel: str(n.getChannelId()),
              title: str(ex.getCharSequence('android.title')), text: str(ex.getCharSequence('android.text')),
              big_text: str(ex.getCharSequence('android.bigText')), sub_text: str(ex.getCharSequence('android.subText')),
              conversation: str(ex.getCharSequence('android.conversationTitle')),
              template: str(ex.getString('android.template')),
              is_group: !!ex.getBoolean('android.isGroupConversation')});
      } catch (e) { send({kind: 'error', msg: 'hook body: ' + e}); }
      return ov.apply(this, arguments);  // never alter or swallow the original call
    };
    hooked++;
  });
  send({kind: 'hooked', count: hooked});
});

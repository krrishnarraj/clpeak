/// The route a saved run's results open on.
///
/// The shell watches what is pushed over its tabs: a route named this way
/// shows a saved run, which the sidebar marks among the recent runs and
/// remembers as opened (`SettingsService.noteRunOpened`).
library;

import 'package:flutter/widgets.dart';

const _prefix = '/run/';

/// The settings of a route that shows the saved run [id].
RouteSettings savedRunRoute(String id) => RouteSettings(name: '$_prefix$id');

/// The saved run [route] shows, or null for any other route.
String? savedRunOf(Route<dynamic> route) {
  final name = route.settings.name;
  return name != null && name.startsWith(_prefix)
      ? name.substring(_prefix.length)
      : null;
}

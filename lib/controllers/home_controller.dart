import 'package:flutter/material.dart';
import 'package:get/get.dart';

class HomeController extends GetxController {
  final currentTab = 0.obs;

  /// Root scaffold key — the chat sidebar hangs off the root so it slides
  /// over the bottom navigation bar instead of under it.
  final scaffoldKey = GlobalKey<ScaffoldState>();

  void changeTab(int index) {
    currentTab.value = index;
  }

  void openSidebar() => scaffoldKey.currentState?.openDrawer();

  void closeSidebar() => scaffoldKey.currentState?.closeDrawer();
}

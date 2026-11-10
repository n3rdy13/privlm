import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:intl/intl.dart';

import '../controllers/task_controller.dart';
import '../core/ui.dart';
import '../models/task_model.dart';
import '../core/haptics.dart';

class TaskView extends GetView<TaskController> {
  const TaskView({super.key});

  @override
  Widget build(BuildContext context) {
    final t = AppTokens.of(context);
    return Obx(() {
      final current = controller.currentTask.value;
      return PopScope(
        canPop: current == null,
        onPopInvokedWithResult: (didPop, _) {
          if (!didPop) controller.currentTask.value = null;
        },
        child: Scaffold(
          backgroundColor: t.bg,
          appBar: current == null
              ? appScreenBar(
                  context,
                  title: 'Tasks',
                  showBack: Navigator.of(context).canPop(),
                  actions: [
                    AppBarIcon(
                      icon: Icons.add_rounded,
                      tooltip: 'New task',
                      onPressed: () => _showCreateSheet(context),
                    ),
                  ],
                )
              : appScreenBar(
                  context,
                  title: 'Task',
                  leading: IconButton(
                    tooltip: 'All tasks',
                    icon: Icon(Icons.arrow_back_rounded, color: t.text),
                    onPressed: () => controller.currentTask.value = null,
                  ),
                  actions: [
                    AppBarIcon(
                      icon: Icons.delete_outline_rounded,
                      tooltip: 'Delete task',
                      onPressed: controller.isExecuting.value
                          ? null
                          : () => _confirmDelete(context, current),
                    ),
                  ],
                ),
          body: current == null
              ? _buildTaskList(context)
              : _buildTaskDetail(context, current),
        ),
      );
    });
  }

  // ── List ──────────────────────────────────────────────────────────────

  Widget _buildTaskList(BuildContext context) {
    if (controller.tasks.isEmpty) {
      return Center(
        child: SingleChildScrollView(
          child: EmptyState(
            icon: Icons.checklist_rounded,
            title: 'No tasks yet',
            message:
                'Describe a goal and the model will plan it\ninto steps, then run them for you.',
            action: PrimaryButton(
              label: 'New task',
              icon: Icons.add_rounded,
              expand: false,
              onPressed: () => _showCreateSheet(context),
            ),
          ),
        ),
      );
    }

    return ListView(
      padding: kPagePadding,
      children: [
        SectionHeader(
          'Recent',
          padding: const EdgeInsets.fromLTRB(4, 12, 4, 8),
          trailing: Text('${controller.tasks.length}',
              style: AppTokens.of(context).small),
        ),
        AppGroup(children: [
          for (final task in controller.tasks) _TaskRow(task: task),
        ]),
        const SectionFootnote('Swipe a task left to delete it.'),
      ],
    );
  }

  // ── Detail ────────────────────────────────────────────────────────────

  Widget _buildTaskDetail(BuildContext context, TaskModel task) {
    final t = AppTokens.of(context);
    final steps = task.steps;
    final done = steps.where((s) => s.status == 'done').length;
    final planning = controller.isPlanning.value;
    final executing = controller.isExecuting.value;

    return Column(children: [
      Expanded(
        child: ListView(
          padding: kPagePadding,
          children: [
            const SizedBox(height: 4),
            Text(task.goal,
                style: t.title.copyWith(fontSize: 20, letterSpacing: -0.5)),
            const SizedBox(height: 10),
            Row(children: [
              _statusTag(context, task.status),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  steps.isEmpty
                      ? _formatDate(task.createdAt)
                      : '$done of ${steps.length} steps · ${_formatDate(task.createdAt)}',
                  style: t.small,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ]),
            if (steps.isNotEmpty) ...[
              const SizedBox(height: 14),
              AppProgressBar(
                value: done / steps.length,
                color: task.status == 'failed' ? t.red : t.green,
              ),
            ],
            if (planning)
              Padding(
                padding: const EdgeInsets.only(top: 48),
                child: Column(children: [
                  SizedBox(
                    width: 22,
                    height: 22,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: t.muted),
                  ),
                  const SizedBox(height: 14),
                  Text('Planning steps…', style: t.caption),
                ]),
              )
            else if (steps.isEmpty)
              const EmptyState(
                icon: Icons.format_list_bulleted_rounded,
                title: 'No steps generated',
                message: 'The model did not return a plan for this goal.',
              )
            else ...[
              const SectionHeader('Steps'),
              AppGroup(
                dividerIndent: 48,
                children: [
                  for (var i = 0; i < steps.length; i++)
                    _StepRow(step: steps[i], number: i + 1),
                ],
              ),
            ],
          ],
        ),
      ),

      // ── Bottom action ──
      if (steps.isNotEmpty && task.status != 'planning')
        SafeArea(
          top: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            child: task.status == 'completed'
                ? AppNotice('Task completed',
                    icon: Icons.check_circle_outline_rounded, color: t.green)
                : PrimaryButton(
                    label: executing ? 'Running…' : 'Run all steps',
                    icon: Icons.play_arrow_rounded,
                    busy: executing,
                    onPressed: () {
                      Haptics.mediumImpact();
                      controller.executeTask(task);
                    },
                  ),
          ),
        ),
    ]);
  }

  // ── Sheets ────────────────────────────────────────────────────────────

  void _showCreateSheet(BuildContext context) {
    final textCtrl = TextEditingController();
    showAppSheet<void>(
      context,
      title: 'New task',
      subtitle: 'The model will break your goal into steps you can run.',
      builder: (ctx) {
        void submit() {
          final goal = textCtrl.text.trim();
          if (goal.isNotEmpty) controller.createTask(goal);
          Navigator.pop(ctx);
        }

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            AppField(
              label: 'Goal',
              child: TextField(
                controller: textCtrl,
                autofocus: true,
                minLines: 3,
                maxLines: 6,
                textCapitalization: TextCapitalization.sentences,
                style: AppTokens.of(ctx).body,
                decoration: appInputDecoration(ctx,
                    hint: 'Describe what you want done…'),
              ),
            ),
            const SizedBox(height: 18),
            ValueListenableBuilder<TextEditingValue>(
              valueListenable: textCtrl,
              builder: (_, value, __) => PrimaryButton(
                label: 'Create task',
                onPressed: value.text.trim().isEmpty ? null : submit,
              ),
            ),
          ],
        );
      },
    ).whenComplete(textCtrl.dispose);
  }

  Future<void> _confirmDelete(BuildContext context, TaskModel task) async {
    final ok = await confirmSheet(
      context,
      title: 'Delete task?',
      message: 'This removes the task and its steps.',
      confirmLabel: 'Delete',
      destructive: true,
    );
    if (!ok) return;
    controller.currentTask.value = null;
    controller.deleteTask(task.id);
  }
}

// ── Shared helpers ──────────────────────────────────────────────────────

String _formatDate(DateTime d) {
  final now = DateTime.now();
  final sameDay =
      d.year == now.year && d.month == now.month && d.day == now.day;
  if (sameDay) return 'Today, ${DateFormat.jm().format(d)}';
  if (d.year == now.year) return DateFormat.MMMd().format(d);
  return DateFormat.yMMMd().format(d);
}

Widget _statusTag(BuildContext context, String status) {
  final t = AppTokens.of(context);
  switch (status) {
    case 'planning':
      return const AppTag('Planning');
    case 'running':
      return AppTag('Running', color: t.accent);
    case 'completed':
      return AppTag('Completed', color: t.green);
    case 'failed':
      return AppTag('Failed', color: t.red);
    default:
      return const AppTag('Ready');
  }
}

class _TaskRow extends GetView<TaskController> {
  final TaskModel task;
  const _TaskRow({required this.task});

  @override
  Widget build(BuildContext context) {
    final t = AppTokens.of(context);
    final steps = task.steps.length;
    return Dismissible(
      key: ValueKey(task.id),
      direction: DismissDirection.endToStart,
      background: Container(
        color: t.red,
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.only(right: 20),
        child: const Icon(Icons.delete_outline_rounded, color: Colors.white),
      ),
      onDismissed: (_) => controller.deleteTask(task.id),
      child: InkWell(
        onTap: () {
          Haptics.selectionClick();
          controller.currentTask.value = task;
        },
        splashFactory: NoSplash.splashFactory,
        highlightColor: t.surfacePressed,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 10, 12),
          child: Row(children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(task.goal,
                      style: t.body,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 6),
                  Row(children: [
                    _statusTag(context, task.status),
                    const SizedBox(width: 8),
                    Flexible(
                      child: Text(
                        '$steps ${steps == 1 ? 'step' : 'steps'} · ${_formatDate(task.createdAt)}',
                        style: t.small,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ]),
                ],
              ),
            ),
            const SizedBox(width: 6),
            Icon(Icons.chevron_right_rounded, size: 20, color: t.faint),
          ]),
        ),
      ),
    );
  }
}

class _StepRow extends StatelessWidget {
  final TaskStep step;
  final int number;
  const _StepRow({required this.step, required this.number});

  @override
  Widget build(BuildContext context) {
    final t = AppTokens.of(context);
    final failed = step.status == 'failed';
    final output = step.output;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 13, 16, 13),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Padding(
          padding: const EdgeInsets.only(top: 1),
          child: SizedBox(
              width: 20, height: 20, child: _statusIcon(t, step.status)),
        ),
        const SizedBox(width: 12),
        Expanded(
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(
              step.description,
              style: t.body.copyWith(
                fontSize: 14.5,
                height: 1.35,
                fontWeight: step.status == 'running'
                    ? FontWeight.w600
                    : FontWeight.w400,
              ),
            ),
            if (step.command != null) ...[
              const SizedBox(height: 8),
              Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: t.isDark
                      ? const Color(0xFF0E0E10)
                      : const Color(0xFFF0F0F2),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: t.hairline, width: 0.5),
                ),
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
                  child: Text(step.command!, style: t.mono),
                ),
              ),
            ],
            if (output != null && output.isNotEmpty) ...[
              const SizedBox(height: 6),
              Text(
                output,
                style: t.caption.copyWith(color: failed ? t.red : t.muted),
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ]),
        ),
        const SizedBox(width: 8),
        Text('$number', style: t.small.copyWith(color: t.faint)),
      ]),
    );
  }

  Widget _statusIcon(AppTokens t, String status) {
    switch (status) {
      case 'running':
        return Padding(
          padding: const EdgeInsets.all(2),
          child: CircularProgressIndicator(strokeWidth: 2, color: t.accent),
        );
      case 'done':
        return Icon(Icons.check_circle_rounded, size: 20, color: t.green);
      case 'failed':
        return Icon(Icons.error_rounded, size: 20, color: t.red);
      default:
        return Icon(Icons.circle_outlined, size: 20, color: t.faint);
    }
  }
}

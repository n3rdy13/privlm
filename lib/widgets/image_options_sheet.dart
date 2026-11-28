import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:image_picker/image_picker.dart';

import '../core/ui.dart';
import '../ffi/sd_ffi_bindings.dart';
import '../services/image_options_service.dart';
import '../services/local_image_service.dart';

/// Image options for the chat: what to avoid, a starting image
/// (image-to-image) and LoRAs.
Future<void> showImageOptionsSheet(BuildContext context) {
  return showAppSheet<void>(
    context,
    title: 'Image options',
    subtitle: 'Fine-tune the next images you make in the chat.',
    builder: (ctx) => const _ImageOptionsBody(),
  );
}

class _ImageOptionsBody extends StatefulWidget {
  const _ImageOptionsBody();

  @override
  State<_ImageOptionsBody> createState() => _ImageOptionsBodyState();
}

class _ImageOptionsBodyState extends State<_ImageOptionsBody> {
  final ImageOptionsService opts = Get.find<ImageOptionsService>();
  late final TextEditingController _negative = TextEditingController(text: opts.negative.value);
  bool _adding = false;

  @override
  void dispose() {
    _negative.dispose();
    super.dispose();
  }

  Future<void> _addLora() async {
    setState(() => _adding = true);
    try {
      final error = await opts.addLora();
      if (error != null) showToast('LoRA not added', error, tone: ToastTone.warning);
    } catch (e) {
      showToast('LoRA not added', '$e', tone: ToastTone.error);
    } finally {
      if (mounted) setState(() => _adding = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = AppTokens.of(context);
    final local = Get.find<LocalImageService>();
    return Obx(() {
      final coreMl = GetPlatform.isIOS && local.currentBackend.value == Backend.metal;
      final init = opts.initImage.value;
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          if (coreMl) ...[
            AppNotice(
              'Starting images and LoRAs need the CPU image backend, not GPU '
              '(Apple NPU). Switch Backend to CPU in Settings → Image generation, '
              'then load the image model again.',
              icon: Icons.info_outline_rounded,
              color: t.orange,
            ),
            const SizedBox(height: 12),
          ],

          // Avoid
          AppField(
            label: 'Avoid',
            helper: 'Things to keep out of the image, e.g. blurry, extra fingers, text.',
            child: TextField(
              controller: _negative,
              maxLines: 2,
              minLines: 1,
              decoration: appInputDecoration(context, hint: 'blurry, deformed, watermark'),
              onChanged: (v) {
                opts.negative.value = v;
                opts.save();
              },
            ),
          ),
          const SizedBox(height: 18),

          // Starting image
          Text('Start from an image', style: t.bodyStrong),
          const SizedBox(height: 4),
          Text(
            'Your prompt then changes this picture instead of starting from nothing.',
            style: t.caption,
          ),
          const SizedBox(height: 10),
          Row(children: [
            Container(
              width: 64,
              height: 64,
              clipBehavior: Clip.antiAlias,
              decoration: BoxDecoration(
                color: t.surfacePressed,
                borderRadius: BorderRadius.circular(12),
              ),
              child: init == null
                  ? Icon(Icons.image_outlined, color: t.faint)
                  : Image.memory(init, fit: BoxFit.cover),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Wrap(spacing: 8, runSpacing: 8, children: [
                PillButton(
                  label: 'Photos',
                  icon: Icons.photo_library_outlined,
                  onPressed: () => opts.pickInitImage(ImageSource.gallery),
                ),
                PillButton(
                  label: 'Camera',
                  icon: Icons.photo_camera_outlined,
                  onPressed: () => opts.pickInitImage(ImageSource.camera),
                ),
                if (init != null)
                  PillButton(
                    label: 'Remove',
                    icon: Icons.close_rounded,
                    onPressed: opts.clearInitImage,
                  ),
              ]),
            ),
          ]),
          if (init != null) ...[
            const SizedBox(height: 10),
            Row(children: [
              Text('Change', style: t.caption),
              Expanded(
                child: Slider(
                  value: opts.strength.value.clamp(0.1, 1.0),
                  min: 0.1,
                  max: 1.0,
                  divisions: 18,
                  label: opts.strength.value.toStringAsFixed(2),
                  onChanged: (v) => opts.strength.value = v,
                  onChangeEnd: (_) => opts.save(),
                ),
              ),
              SizedBox(
                width: 36,
                child: Text(opts.strength.value.toStringAsFixed(2), style: t.caption),
              ),
            ]),
            Text(
              'A little (0.3) keeps the picture and changes details or style. '
              'A lot (0.7+) keeps only its rough layout and colors.',
              style: t.caption,
            ),
          ],
          const SizedBox(height: 18),

          // LoRAs
          Row(children: [
            Expanded(child: Text('LoRAs', style: t.bodyStrong)),
            PillButton(
              label: _adding ? 'Adding…' : 'Add LoRA',
              icon: Icons.add_rounded,
              onPressed: _adding ? null : _addLora,
            ),
          ]),
          const SizedBox(height: 4),
          Text(
            'A LoRA adds a style, character or detail. Use ones made for your '
            'image model (SD 1.5 LoRAs for SD 1.5 models). Many need a trigger '
            'word in the prompt.',
            style: t.caption,
          ),
          const SizedBox(height: 8),
          if (opts.loras.isEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: Text('No LoRAs yet. Tap Add LoRA and pick a .safetensors file.',
                  style: t.caption),
            ),
          for (final lora in opts.loras) _LoraTile(lora: lora, opts: opts),
          const SizedBox(height: 8),
        ],
      );
    });
  }
}

class _LoraTile extends StatelessWidget {
  final LoraEntry lora;
  final ImageOptionsService opts;
  const _LoraTile({required this.lora, required this.opts});

  @override
  Widget build(BuildContext context) {
    final t = AppTokens.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: AppCard(
        padding: const EdgeInsets.fromLTRB(12, 6, 4, 6),
        child: Column(children: [
          Row(children: [
            Expanded(
              child: Text(lora.name, style: t.body, overflow: TextOverflow.ellipsis),
            ),
            Switch(
              value: lora.use,
              onChanged: (v) {
                lora.use = v;
                opts.save();
              },
            ),
            IconButton(
              icon: Icon(Icons.delete_outline_rounded, color: t.faint, size: 20),
              tooltip: 'Remove',
              onPressed: () => opts.removeLora(lora),
            ),
          ]),
          if (lora.use)
            Row(children: [
              Text('Weight', style: t.caption),
              Expanded(
                child: Slider(
                  value: lora.weight.clamp(-1.0, 2.0),
                  min: -1.0,
                  max: 2.0,
                  divisions: 60,
                  label: lora.weight.toStringAsFixed(2),
                  onChanged: (v) {
                    lora.weight = v;
                    opts.loras.refresh();
                  },
                  onChangeEnd: (_) => opts.save(),
                ),
              ),
              SizedBox(
                width: 40,
                child: Text(lora.weight.toStringAsFixed(2), style: t.caption),
              ),
            ]),
        ]),
      ),
    );
  }
}

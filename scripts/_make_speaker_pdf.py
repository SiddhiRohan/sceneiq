from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT

OUT = r"C:\Users\siddh\Downloads\sceneiq\SceneIQ_Speaker_Script.pdf"

styles = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceAfter=6, textColor="#1f3a8a")
h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=11, spaceAfter=4)
body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=8, alignment=TA_LEFT)
q = ParagraphStyle("q", parent=body, textColor="#1f3a8a", spaceAfter=2, fontName="Helvetica-Bold")

doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.8*inch, rightMargin=0.8*inch, topMargin=0.8*inch, bottomMargin=0.8*inch)
story = []

story.append(Paragraph("SceneIQ — Speaker Script & FAQ", h1))
story.append(Paragraph("MSML640 Project Presentation • ~4:45 min • Split across 3 speakers", body))
story.append(Spacer(1, 10))

# Siddhi
story.append(Paragraph("Siddhi — Slides 1 &amp; 2 (~1:30)", h2))
story.append(Paragraph("<b>[Slide 1]</b> Hi everyone, I'm Siddhi, and with me are Krishna and Dhanush. Our project is called SceneIQ, and it's about teaching a vision model to tell when a photo actually <i>makes sense</i>.", body))
story.append(Paragraph("<b>[Slide 2]</b> So here's the thing that got us interested. If you show any modern vision model a picture of a fireplace sitting on a beach, it will happily tell you &quot;I see a fireplace, I see sand, I see water.&quot; It gets every object right — and still completely misses that the scene is absurd. That gap between recognizing objects and understanding whether they belong together is what SceneIQ is trying to close.", body))
story.append(Paragraph("Our approach has four parts. First, we mine Visual Genome's scene graphs to learn which object pairs and relationships are normal in the real world — essentially building a statistical sense of plausibility. Second, we deliberately break that plausibility: we take a coherent scene, find an &quot;alien&quot; object that has <i>zero</i> co-occurrence with anything already in it, crop it from another image, and paste it in. That gives us labelled incoherent data for free. Third, we fine-tune a Vision Transformer as a binary classifier — coherent versus incoherent. And fourth, we evaluate on held-out real and synthetic data. We're targeting ten thousand coherent and five thousand incoherent examples.", body))

# Krishna
story.append(Paragraph("Krishna — Slides 3 &amp; 4 (~1:30)", h2))
story.append(Paragraph("<b>[Slide 3]</b> Thanks Siddhi. I'll talk about the challenges we've hit and how we're working around them.", body))
story.append(Paragraph("The biggest one is the realism gap. When you paste a cropped object into a scene, you get a hard rectangular seam, and there's a real risk the model learns to detect <i>that</i> instead of actual semantic inconsistency. We're mitigating it by scaling pastes to a fraction of the scene size and randomizing placement, but it's something we have to watch in evaluation.", body))
story.append(Paragraph("The second challenge is label noise. Visual Genome has eighty-two thousand object categories, and a lot of them are free-text junk like &quot;there is a man&quot; or &quot;bed sentence.&quot; If those leak into our alien pool, the labels are meaningless. We added a minimum-frequency filter that drops categories seen fewer than fifty times, and that cuts the pool from 82K down to about 2,500 real categories.", body))
story.append(Paragraph("The third one is just scale — generating five thousand synthetic images took about three and a half hours, and the ten thousand coherent downloads will add another couple of hours. Manageable, but not something we can iterate on casually.", body))
story.append(Paragraph("Stack-wise: Python, PyTorch, HuggingFace for the ViT, and Weights &amp; Biases for tracking. The figure on the slide walks through the pipeline end-to-end.", body))
story.append(Paragraph("<b>[Slide 4]</b> For the fallback plan — if the ViT alone isn't strong enough, we have a few options. We can improve the synthesis with blending or diffusion inpainting so the paste artifacts disappear. We can add a GNN branch over the scene graph and fuse it with the ViT features, which is closer to our original proposal. If scale becomes a problem, we can drop to a curated 1K-by-1K set with heavy augmentation and prioritize a clean evaluation over raw size. And worst case, we swap ViT for CLIP to pull in richer vision-language priors.", body))

# Dhanush
story.append(Paragraph("Dhanush — Slides 5, 6, 7 (~1:30)", h2))
story.append(Paragraph("<b>[Slide 5]</b> I'll cover the references and how they shape what we're doing.", body))
story.append(Paragraph("The first paper is ViCor from ICLR 2024. Zhou and colleagues split visual commonsense into two problems — perceiving what's in the image, and inferring what's <i>implied</i>. Their finding is that VLMs consistently miss the inference step. They'll see a person, they'll see snow, but they won't connect them into a coherent scene judgment. That directly motivated us: if the big pretrained models struggle here, there's room for a focused, task-specific architecture.", body))
story.append(Paragraph("The second paper is Ye et al., CVPR 2023. They show that standard vision-language training data captures what things <i>look like</i> but not how they <i>relate</i> to each other, and they propose injecting commonsense from ConceptNet to fix it. For us, this is the explanation for why off-the-shelf VLMs fail at scene coherence — the training data just doesn't emphasize relational knowledge. Our scene-graph integration is basically a response to the same gap, just from a different angle.", body))
story.append(Paragraph("<b>[Slide 6]</b> On originality — this project doesn't overlap with any other coursework any of us have done. We've looked around, and as far as we can tell no prior work combines all three of our choices: VG co-occurrence for plausibility priors, synthetic alien insertion for supervision, and a ViT-plus-optional-GNN classifier. It's also distinct from the two papers we reference — ViCor probes LLM reasoning, DANCE injects ConceptNet into pretraining. We're doing something different: using scene graphs at the <i>data-generation</i> stage to build a labelled benchmark.", body))
story.append(Paragraph("<b>[Slide 7]</b> Quickly on where we stand: we have the download pipeline, co-occurrence tables, the full five thousand synthetic images, and the training and evaluation scripts all working. What's left is assembling the final splits, training the ViT, and doing the per-alien error analysis. Stretch goals are the GNN branch and better synthesis. Happy to take questions.", body))

story.append(PageBreak())
story.append(Paragraph("Likely Questions &amp; Suggested Answers", h1))

faqs = [
    ("Q1. Your model might just be learning the paste artifacts — how do you know it's learning semantics?",
     "That's the concern we take most seriously. Two things: first, we evaluate with a per-alien-category recall breakdown, so if it's only catching paste seams, recall would be roughly uniform regardless of category — we expect semantic categories to vary. Second, the fallback plan includes diffusion inpainting specifically to remove the seam cue and see whether accuracy holds."),
    ("Q2. Why synthetic data? Wouldn't real incoherent images be more convincing?",
     "Real incoherent images are rare and not labelled at scale. Synthetic generation gives us ground truth and volume. We acknowledge the distribution gap and plan to spot-check on real implausible photos (surreal art, photoshop fails) as a qualitative sanity check."),
    ("Q3. Why Visual Genome instead of MS-COCO or a newer dataset?",
     "VG is the only large-scale dataset with dense object <i>and</i> relationship annotations — that's essential for both co-occurrence mining and the eventual scene-graph branch. COCO has the objects but not the relations."),
    ("Q4. Your zero-co-occurrence criterion — isn't that too strict? Some rare but valid combinations exist.",
     "Yes, that's a real limitation. &quot;Zero&quot; here means &quot;zero in the VG corpus,&quot; not &quot;impossible in reality.&quot; We accept some false positives in training data; if evaluation shows the model over-fires on genuinely rare but valid scenes, we'd relax to a low-percentile threshold."),
    ("Q5. How will you handle images where the alien is tiny or occluded?",
     "We already filter pastes below 32 pixels. Beyond that, small aliens are a genuine hard case — we expect recall to drop for small paste bboxes and will report that as a slice in evaluation."),
    ("Q6. Why ViT-base and not a larger model?",
     "Compute and iteration speed. ViT-base fine-tunes in a few hours per run, which lets us actually iterate. If accuracy plateaus, ViT-large or CLIP are in the fallback plan."),
    ("Q7. How do you prevent train/test leakage between the synthetic set and the real coherent set?",
     "Any VG image used as a destination scene in synthesis is explicitly excluded from the coherent sampling pool. Stratified 70/15/15 split with a fixed seed."),
    ("Q8. Can this generalize beyond Visual Genome's domain?",
     "Honestly, probably not out of the box — VG is heavy on indoor and urban scenes. Cross-domain generalization is a follow-up, not an in-scope claim for this project."),
    ("Q9. Why binary classification instead of localizing the inconsistency?",
     "Simpler supervision, cleaner evaluation, and a binary classifier is a natural first step before tackling localization. Localization is a logical next project."),
    ("Q10. What's your baseline? How do you know ViT is a reasonable choice at all?",
     "We plan a zero-shot CLIP baseline and a simple CNN baseline so we have an apples-to-apples comparison. Without those numbers, any accuracy we report isn't meaningful."),
]
for question, answer in faqs:
    story.append(Paragraph(question, q))
    story.append(Paragraph(answer, body))

doc.build(story)
print(f"Wrote: {OUT}")

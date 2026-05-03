"""
MainWindow — Optical Designer application shell.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QToolBar, QFileDialog,
    QMessageBox, QLabel, QApplication, QInputDialog,
    QListWidget, QListWidgetItem, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QDialog,
)
from PySide6.QtGui import QAction, QColor, QPalette, QKeySequence

import json
import os
import numpy as np

from .model import SystemModel, Element, SurfaceRow, SourceDefinition
from .element_table import ElementTableEditor
from .layout_2d import Layout2DView
from .analysis import SpotDiagramWidget, SystemSummaryWidget
from .optimizer_dock import OptimizerDock
from .rayfan_dock import RayFanDock
from .layout_3d import Layout3DView
from .tolerance_dock import ToleranceDock
from .waveoptics_dock import WaveOpticsDock
from .glass_map_dock import GlassMapDock
from .slider_dock import SliderDock
from .library_dock import LibraryDock
from .zernike_dock import ZernikeDock
from .diagnostics import DiagnosticsDock, DiagnosticsBadge, diag
from .workspace import (
    WorkspaceBar, WorkspaceManager, ManageWorkspaceDialog,
    PinnedDocksDialog, default_dock_titles,
)


class MainWindow(QMainWindow):
    """Optical Designer main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Optical Designer')
        self.setMinimumSize(1200, 700)
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.BottomDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)
        self.setDockNestingEnabled(True)

        # ── Model ──
        self.model = SystemModel()

        # File tracking for the window title.  Set by the load/save
        # paths.  `_dirty` flips True on system_changed and is cleared
        # again by save / load / new.  `_suppress_dirty_marking` is
        # raised during programmatic state changes (load_prescription,
        # restore_state) so they don't immediately mark dirty.
        self._current_path = None
        self._dirty = False
        self._suppress_dirty_marking = False
        # Drag-and-drop file open: handled by dragEnterEvent/dropEvent.
        self.setAcceptDrops(True)

        # ── Central widget: element table ──
        self.element_editor = ElementTableEditor(self.model)
        self.element_editor.setMaximumHeight(350)
        self.setCentralWidget(self.element_editor)

        # ── Docks ──
        self._create_docks()
        self._build_menus()
        # Workspace bar sits above the main toolbar; _build_toolbar runs
        # AFTER so its toolbar lands on a new row underneath.
        self._build_workspace_bar()
        self._build_toolbar()
        self._install_shortcuts()
        self._init_workspaces()

        # Ctrl+K / Ctrl+Shift+P command palette: indexed at click time
        # so any later menu changes are reflected on next open.
        try:
            from .command_palette import install_command_palette
            install_command_palette(self)
        except Exception:
            pass

        # ── Status bar ──
        self.status_label = QLabel('Ready')
        self.statusBar().addWidget(self.status_label, stretch=1)

        # Permanent right-aligned metrics: EFL / BFL / F# / EPD / wavelength.
        # Always visible regardless of which workspace tab is active --
        # users keep an eye on the design's headline numbers without
        # opening System Data.
        self.metric_efl  = QLabel('EFL —')
        self.metric_bfl  = QLabel('BFL —')
        self.metric_fnum = QLabel('f/# —')
        self.metric_epd  = QLabel('EPD —')
        self.metric_wv   = QLabel('λ —')
        for w in (self.metric_efl, self.metric_bfl, self.metric_fnum,
                  self.metric_epd, self.metric_wv):
            w.setStyleSheet('padding: 0 8px;')
            self.statusBar().addPermanentWidget(w)

        # Diagnostics badge on the right of the status bar.
        self.diag_badge = DiagnosticsBadge(self)
        self.diag_badge.clicked.connect(
            lambda: self._show_and_raise(self.diagnostics_dock))
        self.statusBar().addPermanentWidget(self.diag_badge)

        self.model.system_changed.connect(self._update_status)
        self.model.system_changed.connect(self._mark_dirty)
        self._refresh_window_title()
        self._update_status()  # populate metrics with the initial model state
        # Seed the Welcome dock's recent-files list from QSettings.
        try:
            self.welcome_widget.set_recent_files(self._recent_files())
        except Exception:
            pass
        self.model.trace_ready.connect(
            lambda r: self.status_label.setText(
                f'Traced {int(np.sum(r.image_rays.alive))}/{r.image_rays.n_rays} rays'
                if r else 'No trace'))

        # Undo/redo: the toolbar actions are wired later; this signal
        # keeps them enabled/disabled correctly.
        self.model.history_changed.connect(self._on_history_changed)

        # ── Autosave: debounce to 1s after the last change -----------
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(1000)
        self._autosave_timer.timeout.connect(self.model.autosave_session)
        self.model.system_changed.connect(self._autosave_timer.start)

        # Track the last-used insert type so Ctrl+R can repeat it.
        self._last_insert_action = None

        # Remember what the user was doing for Ctrl+R ("Repeat last
        # insert") -- set in each _ins_* method below.
        # Value should be a callable with no arguments.

    def showEvent(self, event):
        """After the window is shown, resize bottom docks to be compact."""
        super().showEvent(event)
        # Give more space to the top docks (layout) and less to the bottom
        try:
            self.resizeDocks(
                [self.layout_dock, self.optimizer_dock],
                [400, 150],
                Qt.Vertical,
            )
        except Exception:
            pass

    def _create_docks(self):
        def dock(title, widget, area, name):
            d = QDockWidget(title, self)
            d.setObjectName(name)
            d.setWidget(widget)
            # Allow floating (pop out) + closing.  Users with big
            # simulations tend to want Wave Optics or Spot Diagram on
            # a separate monitor.
            d.setFeatures(QDockWidget.DockWidgetMovable
                          | QDockWidget.DockWidgetFloatable
                          | QDockWidget.DockWidgetClosable)
            self.addDockWidget(area, d)
            return d

        self.layout_view = Layout2DView(self.model)
        self.layout_dock = dock('2D Layout', self.layout_view, Qt.TopDockWidgetArea, 'layout')

        self.layout3d_view = Layout3DView(self.model)
        self.layout3d_dock = dock('3D Layout', self.layout3d_view, Qt.TopDockWidgetArea, 'layout3d')

        self.spot_widget = SpotDiagramWidget(self.model)
        self.spot_dock = dock('Spot Diagram', self.spot_widget, Qt.RightDockWidgetArea, 'spot')

        self.summary_widget = SystemSummaryWidget(self.model)
        self.summary_dock = dock('System Data', self.summary_widget, Qt.RightDockWidgetArea, 'summary')

        self.rayfan_widget = RayFanDock(self.model)
        self.rayfan_dock = dock('Ray Fan / OPD', self.rayfan_widget, Qt.RightDockWidgetArea, 'rayfan')

        self.glassmap_widget = GlassMapDock(self.model)
        self.glassmap_dock = dock('Glass Map', self.glassmap_widget, Qt.RightDockWidgetArea, 'glassmap')

        self.library_widget = LibraryDock(self.model)
        self.library_dock = dock('Library', self.library_widget, Qt.RightDockWidgetArea, 'library')

        self.optimizer_widget = OptimizerDock(self.model)
        self.optimizer_dock = dock('Optimizer', self.optimizer_widget, Qt.BottomDockWidgetArea, 'optimizer')

        self.slider_widget = SliderDock(self.model)
        self.slider_dock = dock('Sliders', self.slider_widget, Qt.BottomDockWidgetArea, 'sliders')

        self.tolerance_widget = ToleranceDock(self.model)
        self.tolerance_dock = dock('Tolerance', self.tolerance_widget, Qt.BottomDockWidgetArea, 'tolerance')

        self.waveoptics_widget = WaveOpticsDock(self.model)
        self.waveoptics_dock = dock('Wave Optics', self.waveoptics_widget, Qt.BottomDockWidgetArea, 'waveoptics')

        self.zernike_widget = ZernikeDock(self.model)
        self.zernike_dock = dock('Zernike', self.zernike_widget, Qt.BottomDockWidgetArea, 'zernike')

        # Snapshots dock: A/B comparison of named designs.
        from .snapshots_dock import SnapshotsDock
        self.snapshots_widget = SnapshotsDock(self.model)
        self.snapshots_dock = dock('Snapshots', self.snapshots_widget,
                                    Qt.RightDockWidgetArea, 'snapshots')

        # ---- New analysis docks (through-focus, PSF/MTF, sensitivity,
        # interferometry, phase retrieval, field browser).  All float
        # in the bottom-docks tab group with the other analysis tools.
        from .through_focus_dock import ThroughFocusDock
        self.through_focus_widget = ThroughFocusDock(self.model)
        self.through_focus_dock = dock(
            'Through-focus', self.through_focus_widget,
            Qt.BottomDockWidgetArea, 'through_focus')

        from .psf_mtf_dock import PSFMTFDock
        self.psfmtf_widget = PSFMTFDock(self.model)
        self.psfmtf_dock = dock(
            'PSF / MTF', self.psfmtf_widget,
            Qt.BottomDockWidgetArea, 'psfmtf')

        from .sensitivity_dock import SensitivityDock
        self.sensitivity_widget = SensitivityDock(self.model)
        self.sensitivity_dock = dock(
            'Sensitivity', self.sensitivity_widget,
            Qt.BottomDockWidgetArea, 'sensitivity')

        from .interferometry_dock import InterferometryDock
        self.interferometry_widget = InterferometryDock(self.model)
        self.interferometry_dock = dock(
            'Interferometry', self.interferometry_widget,
            Qt.BottomDockWidgetArea, 'interferometry')

        from .phase_retrieval_dock import PhaseRetrievalDock
        self.phase_retrieval_widget = PhaseRetrievalDock(self.model)
        self.phase_retrieval_dock = dock(
            'Phase Retrieval', self.phase_retrieval_widget,
            Qt.BottomDockWidgetArea, 'phase_retrieval')

        from .field_browser_dock import FieldBrowserDock
        self.field_browser_widget = FieldBrowserDock(self.model)
        self.field_browser_dock = dock(
            'Field Browser', self.field_browser_widget,
            Qt.RightDockWidgetArea, 'field_browser')

        # Materials dock: Glass Map + User Library in a single tabbed
        # container.  We keep the individual docks alive (and in the
        # View menu) for users who prefer the old split layout.
        from .materials_dock import MaterialsDock
        self.materials_widget = MaterialsDock(self.model)
        self.materials_dock = dock(
            'Materials', self.materials_widget,
            Qt.RightDockWidgetArea, 'materials')

        # Multi-configuration designer.
        from .multiconfig_dock import MultiConfigDock
        self.multiconfig_widget = MultiConfigDock(self.model)
        self.multiconfig_dock = dock(
            'Multi-Config', self.multiconfig_widget,
            Qt.BottomDockWidgetArea, 'multiconfig')

        # Ghost-path analyzer (double-bounce intensity estimates).
        from .ghost_dock import GhostDock
        self.ghost_widget = GhostDock(self.model)
        self.ghost_dock = dock(
            'Ghost Analysis', self.ghost_widget,
            Qt.BottomDockWidgetArea, 'ghost')

        # Jones-pupil 2x4 visualization for polarization analysis.
        from .jones_pupil_dock import JonesPupilDock
        self.jones_pupil_widget = JonesPupilDock(self.model)
        self.jones_pupil_dock = dock(
            'Jones Pupil', self.jones_pupil_widget,
            Qt.RightDockWidgetArea, 'jones_pupil')

        # Footprint plot — ray bundle outline drawn on each surface.
        from .footprint_dock import FootprintDock
        self.footprint_widget = FootprintDock(self.model)
        self.footprint_dock = dock(
            'Footprint', self.footprint_widget,
            Qt.BottomDockWidgetArea, 'footprint')

        # Distortion plot — chief-ray distortion vs field + grid.
        from .distortion_dock import DistortionDock
        self.distortion_widget = DistortionDock(self.model)
        self.distortion_dock = dock(
            'Distortion', self.distortion_widget,
            Qt.BottomDockWidgetArea, 'distortion')

        # Spot-vs-field — array of spot diagrams across configured fields.
        from .spot_field_dock import SpotFieldDock
        self.spot_field_widget = SpotFieldDock(self.model)
        self.spot_field_dock = dock(
            'Spot vs Field', self.spot_field_widget,
            Qt.BottomDockWidgetArea, 'spot_field')

        # Diagnostics dock: hidden by default; the status-bar badge
        # surfaces when there is something to look at.
        self.diagnostics_widget = DiagnosticsDock()
        self.diagnostics_dock = dock(
            'Diagnostics', self.diagnostics_widget,
            Qt.BottomDockWidgetArea, 'diagnostics')
        self.diagnostics_dock.hide()

        # Embedded Python REPL: model, np, plt, result, wave pre-bound.
        from .repl_dock import ReplDock
        self.repl_widget = ReplDock(self.model)
        self.repl_dock = dock(
            'Python', self.repl_widget,
            Qt.BottomDockWidgetArea, 'repl')
        # Stream wave-optics results into the REPL after each run.
        self.waveoptics_widget.run_finished.connect(
            self.repl_widget.set_wave_result)

        # Welcome panel: empty-state guidance with quick-start actions
        # and a recent-files list.  Default in the Design workspace.
        from .welcome_dock import WelcomeDock
        self.welcome_widget = WelcomeDock()
        self.welcome_dock = dock(
            'Welcome', self.welcome_widget,
            Qt.RightDockWidgetArea, 'welcome')
        self.welcome_widget.open_path_requested.connect(self._open_recent)
        self.welcome_widget.open_demo_requested.connect(self._open_demo)
        self.welcome_widget.insert_singlet_requested.connect(
            self._ins_plano_convex)
        self.welcome_widget.insert_achromat_requested.connect(
            lambda: self._ins_thorlabs('AC254-100-C'))
        self.welcome_widget.browse_library_requested.connect(
            lambda: self._show_and_raise(self.library_dock))
        self.welcome_widget.show_shortcuts_requested.connect(
            self._show_shortcuts)

        # When the wave-optics dock finishes a run, hand the focal-plane
        # field to the Zernike dock so its decompose button operates on
        # a real (unwrapped, reference-sphere-subtracted) OPD map.
        self.waveoptics_widget.run_finished.connect(
            self._populate_zernike_from_waveoptics)

        # Tab groups
        self.tabifyDockWidget(self.spot_dock, self.summary_dock)
        self.tabifyDockWidget(self.summary_dock, self.rayfan_dock)
        self.tabifyDockWidget(self.rayfan_dock, self.glassmap_dock)
        self.tabifyDockWidget(self.glassmap_dock, self.library_dock)
        self.spot_dock.raise_()

        self.tabifyDockWidget(self.layout_dock, self.layout3d_dock)
        self.layout_dock.raise_()

        self.tabifyDockWidget(self.optimizer_dock, self.slider_dock)
        self.tabifyDockWidget(self.slider_dock, self.tolerance_dock)
        self.tabifyDockWidget(self.tolerance_dock, self.waveoptics_dock)
        self.tabifyDockWidget(self.waveoptics_dock, self.zernike_dock)
        self.tabifyDockWidget(self.zernike_dock, self.through_focus_dock)
        self.tabifyDockWidget(self.through_focus_dock, self.psfmtf_dock)
        self.tabifyDockWidget(self.psfmtf_dock, self.sensitivity_dock)
        self.tabifyDockWidget(self.sensitivity_dock, self.interferometry_dock)
        self.tabifyDockWidget(self.interferometry_dock,
                               self.phase_retrieval_dock)
        self.tabifyDockWidget(self.snapshots_dock, self.field_browser_dock)
        self.tabifyDockWidget(self.field_browser_dock, self.materials_dock)
        self.tabifyDockWidget(self.phase_retrieval_dock,
                               self.multiconfig_dock)
        self.optimizer_dock.raise_()

        # When a wave-optics run finishes, auto-populate downstream
        # docks that depend on an exit-pupil field.
        self.waveoptics_widget.run_finished.connect(
            self._populate_downstream_docks)

    def _populate_downstream_docks(self, results):
        """Route the latest wave-optics run into the analysis docks that
        consume an exit-pupil field (through-focus, PSF/MTF).  Zernike
        is handled by the existing _populate_zernike_from_waveoptics."""
        if not results or results.get('error'):
            return
        planes = results.get('planes') or []
        if not planes:
            return
        exit_plane = next(
            (p for p in planes if p.get('label') == 'LensExit'), None) \
            or planes[-1]
        try:
            self.through_focus_widget.set_source_field(
                exit_plane['field'],
                exit_plane.get('dx', results.get('dx')),
                float(results['wavelength']),
                z_center=results.get('bfl_m'))
        except Exception:
            pass
        try:
            import numpy as _np
            self.psfmtf_widget._pupil = _np.asarray(
                exit_plane['field'], dtype=_np.complex128)
            self.psfmtf_widget._dx = float(
                exit_plane.get('dx', results.get('dx')))
            self.psfmtf_widget._wavelength = float(results['wavelength'])
            self.psfmtf_widget._focal_length = float(
                results.get('bfl_m') or 0.0)
        except Exception:
            pass

    def _populate_zernike_from_waveoptics(self, results):
        """Hand the focal-plane field from a wave-optics run to the Zernike
        dock so its decomposition runs on a real (unwrapped, reference-
        sphere-subtracted) OPD map.  Fails loudly through the diagnostics
        sink instead of disappearing silently.
        """
        if not results or results.get('error'):
            return
        planes = results.get('planes') or []
        if not planes:
            return
        # Prefer an explicitly-labeled "Focus" plane; if absent, fall
        # back to whichever saved plane has the largest z (the last
        # plane the user saved, regardless of renaming).
        focus = next((p for p in planes if p.get('label') == 'Focus'), None)
        if focus is None:
            focus = max(planes, key=lambda p: p.get('z', 0.0))
        try:
            wavelength = float(results['wavelength'])
            dx = float(focus.get('dx', results.get('dx', 0.0)))
            aperture = float(self.model.epd_m)
            _, efl, _ = self.model.get_abcd()
            self.zernike_widget.set_field(
                focus['field'], dx, wavelength, aperture,
                focal_length=efl if np.isfinite(efl) else None,
            )
        except Exception as e:
            try:
                from .diagnostics import diag
                diag.report('zernike-populate', e,
                            context='populate from wave-optics')
            except Exception:
                pass

    # ------------------------------------------------------------------
    #  Workspace tab system
    # ------------------------------------------------------------------

    def _build_workspace_bar(self):
        """Create the top-of-window workspace tab strip (no contents
        yet — _init_workspaces fills it after the toolbar exists)."""
        self.workspace_bar = WorkspaceBar(self)
        self.addToolBar(Qt.TopToolBarArea, self.workspace_bar)
        # New row break so the main toolbar lands underneath the tabs.
        self.addToolBarBreak(Qt.TopToolBarArea)

    def _init_workspaces(self):
        """Build the dock registry, the workspace manager, restore any
        persisted workspaces (else load defaults), and apply the active
        workspace so the initial UI is the configured one."""
        # Build the registry from the QDockWidget objectName values that
        # were assigned in _create_docks.
        self._dock_registry = {
            d.objectName(): d
            for d in self.findChildren(QDockWidget)
            if d.objectName()
        }

        self.workspace_mgr = WorkspaceManager(self, self._dock_registry)
        self.workspace_mgr.init_defaults()

        # Restore from QSettings if available.
        try:
            from PySide6.QtCore import QSettings
            s = QSettings('lumenairy', 'OpticalDesigner')
            payload = s.value('workspaces/data')
            if payload:
                self.workspace_mgr.load_json(payload)
        except Exception:
            pass

        # Wire dock visibility tracking: when the user toggles a dock
        # via the View menu or close button, update the current
        # workspace's dock_names so the change is sticky.
        for name, dock in self._dock_registry.items():
            act = dock.toggleViewAction()
            act.toggled.connect(
                lambda visible, n=name:
                    self.workspace_mgr.on_user_toggled_dock(n, bool(visible)))

        # Populate the tab bar and apply the saved/active workspace.
        self.workspace_bar.set_names(
            self.workspace_mgr.titles(),
            current_index=self.workspace_mgr.current_index,
        )
        self.workspace_bar.switched.connect(self._on_workspace_switched)
        self.workspace_bar.add_requested.connect(
            self._on_workspace_add_requested)
        self.workspace_bar.rename_requested.connect(
            self._on_workspace_rename_requested)
        self.workspace_bar.delete_requested.connect(
            self._on_workspace_delete_requested)
        self.workspace_bar.duplicate_requested.connect(
            self._on_workspace_duplicate_requested)
        self.workspace_bar.manage_requested.connect(
            self._on_workspace_manage_requested)

        # Apply once -- this hides docks not in the active workspace.
        self.workspace_mgr.apply_index(self.workspace_mgr.current_index)

        # Optimizer progress: status-bar message + badge on the Optimize tab.
        self.model.optimization_progress.connect(self._on_optimization_progress)
        self.model.optimization_finished.connect(self._on_optimization_finished)

    def _on_workspace_switched(self, idx):
        """User clicked a different tab.  Save the outgoing layout
        and apply the new one."""
        if idx == self.workspace_mgr.current_index:
            return
        # Save the current dock geometry into the OUTGOING workspace
        # before switching -- so positions are preserved per-tab.
        self.workspace_mgr.save_current_layout()
        self.workspace_mgr.apply_index(idx)

    def _on_workspace_add_requested(self):
        name, ok = QInputDialog.getText(
            self, 'New Workspace', 'Workspace name:')
        if not ok:
            return
        idx = self.workspace_mgr.add(name=name or 'New Workspace')
        # Rebuild the tab bar (don't fire switched yet).
        self.workspace_bar.set_names(
            self.workspace_mgr.titles(),
            current_index=self.workspace_mgr.current_index,
        )
        # Open Manage Docks immediately so the user can populate it.
        self._on_workspace_manage_requested(idx)
        # Switch to the new tab.
        self.workspace_mgr.save_current_layout()
        self.workspace_mgr.apply_index(idx)
        self.workspace_bar.set_current_index(idx)

    def _on_workspace_rename_requested(self, idx):
        ws = (self.workspace_mgr.workspaces[idx]
              if 0 <= idx < len(self.workspace_mgr.workspaces) else None)
        if ws is None:
            return
        name, ok = QInputDialog.getText(
            self, 'Rename Workspace', 'New name:', text=ws.name)
        if not ok or not name.strip():
            return
        self.workspace_mgr.rename(idx, name.strip())
        self.workspace_bar.set_names(
            self.workspace_mgr.titles(),
            current_index=self.workspace_mgr.current_index,
        )

    def _on_workspace_duplicate_requested(self, idx):
        new_idx = self.workspace_mgr.duplicate(idx)
        if new_idx < 0:
            return
        # Save current layout to source first so the dup carries it.
        self.workspace_mgr.save_current_layout()
        self.workspace_bar.set_names(
            self.workspace_mgr.titles(),
            current_index=self.workspace_mgr.current_index,
        )
        # Switch into the duplicate.
        self.workspace_mgr.apply_index(new_idx)
        self.workspace_bar.set_current_index(new_idx)

    def _on_workspace_delete_requested(self, idx):
        if len(self.workspace_mgr.workspaces) <= 1:
            QMessageBox.information(
                self, 'Delete Workspace',
                'You need to keep at least one workspace.')
            return
        ws = self.workspace_mgr.workspaces[idx]
        ans = QMessageBox.question(
            self, 'Delete Workspace',
            f'Delete the "{ws.name}" workspace?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self.workspace_mgr.remove(idx)
        self.workspace_bar.set_names(
            self.workspace_mgr.titles(),
            current_index=self.workspace_mgr.current_index,
        )
        self.workspace_mgr.apply_index(self.workspace_mgr.current_index)

    def _on_workspace_manage_requested(self, idx):
        ws = (self.workspace_mgr.workspaces[idx]
              if 0 <= idx < len(self.workspace_mgr.workspaces) else None)
        if ws is None:
            return
        titles = default_dock_titles()
        # Make sure every dock that actually exists is offered, even
        # if it's not in the canonical title map.
        for name in self._dock_registry:
            titles.setdefault(name, name)
        # Drop entries for docks that don't exist on this window.
        titles = {n: t for n, t in titles.items()
                  if n in self._dock_registry}
        dlg = ManageWorkspaceDialog(self, ws, titles)
        if dlg.exec() != QDialog.Accepted:
            return
        new_names = dlg.selected_dock_names()
        self.workspace_mgr.set_dock_names(idx, new_names)
        # If the user just edited the current workspace, re-apply it.
        if idx == self.workspace_mgr.current_index:
            self.workspace_mgr.apply_index(idx)

    def _on_optimization_progress(self, iteration, merit):
        """Show optimization progress in status + Optimize tab badge."""
        self.status_label.setText(
            f'Optimizing — iter {iteration}, merit {merit:.4g}')
        try:
            idx = self.workspace_bar.find_tab('Optimize')
            if idx < 0:
                # Try matching the workspace by name in the manager
                for i, ws in enumerate(self.workspace_mgr.workspaces):
                    if ws.name == 'Optimize':
                        idx = i
                        break
            if idx >= 0:
                self.workspace_bar.set_tab_text(
                    idx, f'Optimize • iter {iteration}')
        except Exception:
            pass

    def _on_optimization_finished(self, success, message):
        """Clear the Optimize tab badge and report the final state."""
        self.status_label.setText(message or
                                  ('Optimization complete' if success
                                   else 'Optimization failed'))
        try:
            for i, ws in enumerate(self.workspace_mgr.workspaces):
                if ws.name == 'Optimize':
                    self.workspace_bar.set_tab_text(i, 'Optimize')
                    break
        except Exception:
            pass

    def _set_compact_mode(self, on):
        """Toggle a screen-real-estate-maximizing layout: hide the menu
        bar and dock title bars.  Workspace bar + main toolbar stay so
        the user can still navigate."""
        self.menuBar().setVisible(not on)
        # Replace each dock's title bar with an empty widget when on,
        # restore the default when off.
        for dock in self._dock_registry.values():
            if on:
                dock.setTitleBarWidget(QWidget())
            else:
                dock.setTitleBarWidget(None)

    def _on_pinned_docks_requested(self):
        """Edit the set of docks pinned across all workspaces."""
        titles = default_dock_titles()
        for name in self._dock_registry:
            titles.setdefault(name, name)
        titles = {n: t for n, t in titles.items()
                  if n in self._dock_registry}
        dlg = PinnedDocksDialog(
            self, set(self.workspace_mgr.pinned_docks), titles)
        if dlg.exec() != QDialog.Accepted:
            return
        new_pinned = dlg.selected_dock_names()
        # Replace the set wholesale.
        self.workspace_mgr.pinned_docks = set(new_pinned)
        # Re-apply the current workspace so the pinned changes take
        # effect immediately.
        self.workspace_mgr.apply_index(self.workspace_mgr.current_index)

    def _export_workspaces(self):
        """Export the current workspace set to a .workspace JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Workspaces',
            'optical_designer.workspace',
            'Workspace JSON (*.workspace *.json)')
        if not path:
            return
        try:
            self.workspace_mgr.save_current_layout()
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.workspace_mgr.to_json())
            self.status_label.setText(f'Exported workspaces to {os.path.basename(path)}')
        except Exception as e:
            diag.report('workspace-export', e, context=path)
            QMessageBox.critical(self, 'Export failed', str(e))

    def _import_workspaces(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Import Workspaces', '',
            'Workspace JSON (*.workspace *.json);;All files (*)')
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = f.read()
            ok = self.workspace_mgr.load_json(payload)
            if not ok:
                QMessageBox.warning(self, 'Import',
                                    'No workspaces found in that file.')
                return
            self.workspace_bar.set_names(
                self.workspace_mgr.titles(),
                current_index=self.workspace_mgr.current_index,
            )
            self.workspace_mgr.apply_index(self.workspace_mgr.current_index)
            self.status_label.setText(
                f'Imported workspaces from {os.path.basename(path)}')
        except Exception as e:
            diag.report('workspace-import', e, context=path)
            QMessageBox.critical(self, 'Import failed', str(e))

    def _reset_workspaces(self):
        ans = QMessageBox.question(
            self, 'Reset Workspaces',
            'Discard all custom workspaces and restore defaults?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self.workspace_mgr.init_defaults()
        self.workspace_bar.set_names(
            self.workspace_mgr.titles(),
            current_index=0,
        )
        self.workspace_mgr.apply_index(0)

    def _save_workspace_preferences(self):
        """Persist the workspace set to QSettings."""
        try:
            from PySide6.QtCore import QSettings
            self.workspace_mgr.save_current_layout()
            s = QSettings('lumenairy', 'OpticalDesigner')
            s.setValue('workspaces/data', self.workspace_mgr.to_json())
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self._save_workspace_preferences()
        except Exception:
            pass
        super().closeEvent(event)

    def _build_menus(self):
        mb = self.menuBar()

        # File
        fm = mb.addMenu('&File')
        fm.addAction('&New', self._new_system).setShortcut('Ctrl+N')
        fm.addAction('&Open Prescription File...', self._load_file).setShortcut('Ctrl+O')
        fm.addAction('Load &Thorlabs Lens...', self._load_thorlabs)
        fm.addSeparator()
        fm.addAction('Save Design (&JSON)...', self._save_json_design).setShortcut('Ctrl+S')
        fm.addAction('Open Design (JSON)...', self._open_json_design)
        fm.addSeparator()
        fm.addAction('&Export Prescription...', self._export_prescription)
        fm.addAction('Export &Python Sim Script...', self._export_python_script)
        fm.addSeparator()
        fm.addAction('&Quit', self.close).setShortcut('Ctrl+Q')

        # Edit (undo/redo)
        em = mb.addMenu('&Edit')
        self.act_undo = QAction('&Undo', self)
        self.act_undo.setShortcut(QKeySequence.Undo)   # Ctrl+Z on most platforms
        self.act_undo.setEnabled(False)
        self.act_undo.triggered.connect(self.model.undo)
        em.addAction(self.act_undo)
        self.act_redo = QAction('&Redo', self)
        self.act_redo.setShortcut(QKeySequence.Redo)   # Ctrl+Y / Ctrl+Shift+Z
        self.act_redo.setEnabled(False)
        self.act_redo.triggered.connect(self.model.redo)
        em.addAction(self.act_redo)
        em.addSeparator()
        em.addAction('Save Snapshot...', self._save_snapshot).setShortcut('Ctrl+B')
        em.addAction('Restore Last Session', self._restore_session)

        # Insert
        im = mb.addMenu('&Insert')
        lm = im.addMenu('&Lens')
        lm.addAction('Plano-Convex Singlet...', self._ins_plano_convex).setShortcut('Ctrl+L')
        lm.addAction('Biconvex Singlet...', self._ins_biconvex)
        lm.addAction('Achromatic Doublet...', self._ins_doublet).setShortcut('Ctrl+Shift+L')
        lm.addSeparator()
        lm.addAction('Cylindrical Lens...', self._ins_cylindrical)
        lm.addAction('Biconic Singlet...', self._ins_biconic)

        mm = im.addMenu('&Mirror')
        mm.addAction('Flat Mirror...', self._ins_flat_mirror).setShortcut('Ctrl+M')
        mm.addAction('Curved Mirror...', self._ins_curved_mirror)

        dm = im.addMenu('&DOE / Diffractive')
        dm.addAction('Microlens Array...', self._ins_mla)
        dm.addAction('Diffraction Grating...', self._ins_grating)
        dm.addAction('Dammann Grating...', self._ins_dammann)

        im.addSeparator()
        # Thorlabs catalog grouped by family.  Within each family,
        # entries are sorted by focal length and the EFL is shown in
        # the menu label so users can pick by f rather than part #.
        cm = im.addMenu('Thorlabs &Catalog')
        self._build_thorlabs_menu(cm)

        im.addSeparator()
        im.addAction('Repeat last insert', self._repeat_last_insert).setShortcut('Ctrl+R')
        im.addAction('&Delete Element', self._delete_element).setShortcut('Ctrl+D')
        im.addSeparator()
        im.addAction('Find nearest Thorlabs part (from current system EFL)...',
                     self._find_nearest_thorlabs)

        # Analysis: raise/focus the corresponding dock rather than
        # spawning ad-hoc dialogs so every feature has a single
        # canonical entry point.
        am = mb.addMenu('&Analysis')
        am.addAction('&Retrace', lambda: self.model.run_trace()).setShortcut('Ctrl+T')
        am.addSeparator()
        am.addAction('Through-focus scan',
                     lambda: self._show_and_raise(self.through_focus_dock))
        am.addAction('PSF / MTF',
                     lambda: self._show_and_raise(self.psfmtf_dock))
        am.addAction('Zernike decomposition',
                     lambda: self._show_and_raise(self.zernike_dock))
        am.addAction('Sensitivity',
                     lambda: self._show_and_raise(self.sensitivity_dock))
        am.addAction('Interferometry',
                     lambda: self._show_and_raise(self.interferometry_dock))
        am.addAction('Phase retrieval',
                     lambda: self._show_and_raise(self.phase_retrieval_dock))
        am.addAction('Field browser (HDF5/Zarr)',
                     lambda: self._show_and_raise(self.field_browser_dock))
        am.addAction('Ghost analysis',
                     lambda: self._show_and_raise(self.ghost_dock))
        am.addAction('Jones pupil',
                     lambda: self._show_and_raise(self.jones_pupil_dock))
        am.addAction('Footprint',
                     lambda: self._show_and_raise(self.footprint_dock))
        am.addAction('Distortion',
                     lambda: self._show_and_raise(self.distortion_dock))
        am.addAction('Spot vs Field',
                     lambda: self._show_and_raise(self.spot_field_dock))
        am.addSeparator()
        am.addAction('Chromatic focal shift (dialog)', self._run_chromatic)
        am.addAction('Quick Zernikes from ray-trace OPD',
                     self._zernike_from_trace)
        am.addSeparator()
        am.addAction('Run Wave Optics (F5)', self._run_waveoptics_now)\
            .setShortcut('F5')
        am.addSeparator()
        am.addAction('Export current spot diagram (PNG)...',
                     self._export_spot_png)
        am.addAction('Export design report (HTML)...',
                     self._export_report_html)

        # View
        vm = mb.addMenu('&View')
        for d in [self.layout_dock, self.layout3d_dock, self.spot_dock,
                  self.rayfan_dock, self.summary_dock, self.glassmap_dock,
                  self.library_dock, self.snapshots_dock,
                  self.field_browser_dock, self.materials_dock,
                  self.jones_pupil_dock]:
            vm.addAction(d.toggleViewAction())
        vm.addSeparator()
        for d in [self.optimizer_dock, self.slider_dock,
                  self.tolerance_dock, self.waveoptics_dock,
                  self.zernike_dock, self.through_focus_dock,
                  self.psfmtf_dock, self.sensitivity_dock,
                  self.interferometry_dock, self.phase_retrieval_dock,
                  self.multiconfig_dock, self.ghost_dock,
                  self.footprint_dock, self.distortion_dock,
                  self.spot_field_dock,
                  self.diagnostics_dock]:
            vm.addAction(d.toggleViewAction())

        vm.addSeparator()
        ws_menu = vm.addMenu('&Workspace')
        ws_menu.addAction('New Workspace...', self._on_workspace_add_requested)
        ws_menu.addAction('Rename Current...',
                          lambda: self._on_workspace_rename_requested(
                              self.workspace_mgr.current_index))
        ws_menu.addAction('Duplicate Current',
                          lambda: self._on_workspace_duplicate_requested(
                              self.workspace_mgr.current_index))
        ws_menu.addAction('Delete Current',
                          lambda: self._on_workspace_delete_requested(
                              self.workspace_mgr.current_index))
        ws_menu.addSeparator()
        ws_menu.addAction('Manage Docks for Current...',
                          lambda: self._on_workspace_manage_requested(
                              self.workspace_mgr.current_index))
        ws_menu.addAction('Pin Docks Across All Workspaces...',
                          self._on_pinned_docks_requested)
        ws_menu.addSeparator()
        ws_menu.addAction('Export Workspaces to File...',
                          self._export_workspaces)
        ws_menu.addAction('Import Workspaces from File...',
                          self._import_workspaces)
        ws_menu.addSeparator()
        ws_menu.addAction('Reset Workspaces to Defaults',
                          self._reset_workspaces)

        vm.addSeparator()
        compact_act = QAction('&Compact Mode', self, checkable=True)
        compact_act.setShortcut('F11')
        compact_act.setToolTip(
            'Hide menu bar and dock title bars to maximize working area '
            '(F11 to toggle back).')
        compact_act.toggled.connect(self._set_compact_mode)
        vm.addAction(compact_act)
        self._compact_action = compact_act

        vm.addSeparator()
        colors_menu = vm.addMenu('&Colors')
        colors_menu.addAction('2D Layout Background...', self._pick_bg_2d)
        colors_menu.addAction('3D Layout Background...', self._pick_bg_3d)
        colors_menu.addAction('Ray Color...', self._pick_ray_color)
        colors_menu.addAction('Use Wavelength Colors',
                              self._toggle_wavelength_colors).setCheckable(True)
        colors_menu.addSeparator()
        colors_menu.addAction('UI Highlight Color...', self._pick_accent)

        vm.addSeparator()
        theme_menu = vm.addMenu('&Theme')
        theme_menu.addAction('Dark', lambda: self._set_theme('dark'))
        theme_menu.addAction('Light', lambda: self._set_theme('light'))
        theme_menu.addAction('Midnight Blue', lambda: self._set_theme('midnight'))

        # Preferences (between Theme and Help).
        pm = mb.addMenu('&Preferences')
        units_menu = pm.addMenu('&Units')
        self._units_group = []
        for name, key in [('Engineering (mm / um / nm)', 'engineering'),
                           ('SI (m / m / m)',            'si')]:
            act = QAction(name, self, checkable=True)
            act.setChecked(self.model.unit_preference == key)
            act.triggered.connect(lambda _=False, k=key: self._set_units(k))
            units_menu.addAction(act)
            self._units_group.append((key, act))

        retrace_menu = pm.addMenu('Auto-&retrace')
        self._retrace_group = []
        for name, key in [('On every edit (default)', 'on'),
                           ('Geometric only',          'geometric-only'),
                           ('Manual (Ctrl+T)',         'manual')]:
            act = QAction(name, self, checkable=True)
            act.setChecked(self.model.auto_retrace_mode == key)
            act.triggered.connect(
                lambda _=False, k=key: self._set_retrace_mode(k))
            retrace_menu.addAction(act)
            self._retrace_group.append((key, act))

        pm.addSeparator()
        pm.addAction('Report error-routing policy...',
                     self._edit_error_routing_policy)

        # Options — per-function kwarg overrides, surfaced from the
        # core library through a tabbed dialog.
        om = mb.addMenu('&Options')
        om.addAction('Lens function options...',
                     self._show_lens_options)

        # Help
        hm = mb.addMenu('&Help')
        cmd_act = QAction('Command Palette (Ctrl+K)...', self)
        cmd_act.setShortcut('Ctrl+K')
        cmd_act.triggered.connect(self._open_command_palette)
        hm.addAction(cmd_act)
        hm.addAction('Keyboard Shortcuts', self._show_shortcuts)
        hm.addAction('&About', self._show_about)

    # ------------------------------------------------------------------
    # Preference handlers
    # ------------------------------------------------------------------

    def _set_units(self, key):
        self.model.unit_preference = key
        for k, act in getattr(self, '_units_group', []):
            act.setChecked(k == key)
        # Notify the element table so it repaints number suffixes.
        try:
            self.model.display_changed.emit()
        except Exception:
            pass
        self.status_label.setText(f'Unit preference: {key}')

    def _set_retrace_mode(self, key):
        self.model.auto_retrace_mode = key
        for k, act in getattr(self, '_retrace_group', []):
            act.setChecked(k == key)
        self.status_label.setText(f'Auto-retrace: {key}')

    def _edit_error_routing_policy(self):
        """Toggle the canonical error-routing policy (used by the
        diagnostics sink): where do non-fatal errors surface?"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QLabel
        try:
            from .diagnostics import diag
        except Exception:
            QMessageBox.warning(
                self, 'Unavailable', 'Diagnostics sink not present.')
            return
        dlg = QDialog(self)
        dlg.setWindowTitle('Error-routing policy')
        dlg.setMinimumWidth(360)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(
            'Choose where each severity surfaces.  The Diagnostics '
            'dock always gets everything; these toggles control '
            'whether higher-severity events also pop a modal dialog '
            'or flash the status bar.'))
        chk_modal_error = QCheckBox('Pop modal on ERROR')
        chk_modal_error.setChecked(
            getattr(diag, 'modal_on_error', True))
        layout.addWidget(chk_modal_error)
        chk_status_warn = QCheckBox('Flash status bar on WARN')
        chk_status_warn.setChecked(
            getattr(diag, 'status_on_warn', True))
        layout.addWidget(chk_status_warn)
        from PySide6.QtWidgets import QDialogButtonBox
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        layout.addWidget(bb)
        if dlg.exec() == QDialog.Accepted:
            diag.modal_on_error = chk_modal_error.isChecked()
            diag.status_on_warn = chk_status_warn.isChecked()

    # ------------------------------------------------------------------
    # Report export
    # ------------------------------------------------------------------

    def _export_report_html(self):
        """One-page HTML report: layout, spot, fans, Zernikes, summary.

        Images are embedded as base64 PNGs so the file is self-contained.
        """
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export design report',
            filter='HTML (*.html)')
        if not path:
            return
        try:
            import base64
            from io import BytesIO
            import numpy as np
            parts = []
            parts.append(
                '<!doctype html><html><head><meta charset="utf-8">'
                '<title>Optical design report</title>'
                '<style>'
                'body{font-family:sans-serif;max-width:980px;margin:2em auto;'
                'color:#222}'
                'h1{border-bottom:2px solid #5cb8ff}'
                'h2{color:#3a6a9a;margin-top:1.5em}'
                'table{border-collapse:collapse;margin:0.5em 0}'
                'td,th{padding:2px 10px;border:1px solid #ccc;font-family:monospace}'
                '.fig{margin:0.5em 0}'
                '</style></head><body>')
            parts.append(f'<h1>Optical design report</h1>')
            try:
                from datetime import datetime
                parts.append(f'<p>Generated {datetime.now().isoformat(timespec="seconds")}</p>')
            except Exception:
                pass

            # System summary
            m = self.model
            parts.append('<h2>System summary</h2>')
            _, efl, bfl = m.get_abcd()
            parts.append('<table>')
            for lbl, val in [
                ('Wavelength (nm)', f'{m.wavelength_nm:.2f}'),
                ('EPD (mm)',         f'{m.epd_mm:.3f}'),
                ('EFL (mm)',         f'{efl*1e3:.4f}' if np.isfinite(efl) else 'inf'),
                ('BFL (mm)',         f'{bfl*1e3:.4f}' if np.isfinite(bfl) else 'inf'),
                ('Elements',         str(len(m.elements))),
            ]:
                parts.append(f'<tr><th>{lbl}</th><td>{val}</td></tr>')
            parts.append('</table>')

            def _fig_to_b64(fig):
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                return base64.b64encode(buf.getvalue()).decode('ascii')

            # Layout
            try:
                if hasattr(self, 'layout_view') and hasattr(self.layout_view,
                                                             'fig'):
                    parts.append('<h2>Layout</h2>')
                    parts.append(
                        f'<div class="fig"><img src="data:image/png;base64,'
                        f'{_fig_to_b64(self.layout_view.fig)}"/></div>')
            except Exception:
                pass
            # Spot diagram
            try:
                if hasattr(self, 'spot_widget'):
                    parts.append('<h2>Spot diagram</h2>')
                    # Grab widget pixels via Qt's QBuffer/QByteArray for
                    # PNG serialisation -- no matplotlib figure exists
                    # for the SpotDiagramWidget.
                    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
                    pm = self.spot_widget.grab()
                    ba = QByteArray()
                    qb = QBuffer(ba)
                    qb.open(QIODevice.WriteOnly)
                    pm.save(qb, 'PNG')
                    qb.close()
                    b64 = base64.b64encode(bytes(ba)).decode('ascii')
                    parts.append(
                        f'<div class="fig"><img src="data:image/png;base64,'
                        f'{b64}"/></div>')
            except Exception:
                pass
            # Ray fan
            try:
                if hasattr(self.rayfan_widget, 'fig'):
                    parts.append('<h2>Ray fan / OPD</h2>')
                    parts.append(
                        f'<div class="fig"><img src="data:image/png;base64,'
                        f'{_fig_to_b64(self.rayfan_widget.fig)}"/></div>')
            except Exception:
                pass
            # Zernikes
            try:
                if hasattr(self.zernike_widget, 'fig') and \
                   self.zernike_widget._coeffs is not None:
                    parts.append('<h2>Zernike decomposition</h2>')
                    parts.append(
                        f'<div class="fig"><img src="data:image/png;base64,'
                        f'{_fig_to_b64(self.zernike_widget.fig)}"/></div>')
            except Exception:
                pass

            # Prescription table
            try:
                pres = self.model.to_prescription()
                parts.append('<h2>Prescription</h2>')
                parts.append('<table><tr><th>Surf</th><th>radius</th>'
                             '<th>thickness</th><th>glass</th><th>conic</th></tr>')
                thk = pres.get('thicknesses', [])
                for i, s in enumerate(pres.get('surfaces', [])):
                    R = s.get('radius', float('inf'))
                    k = s.get('conic', 0.0)
                    g = s.get('glass_after', '') or ''
                    t = thk[i] if i < len(thk) else 0.0
                    R_txt = 'inf' if not np.isfinite(R) else f'{R:.6g}'
                    parts.append(f'<tr><td>{i}</td><td>{R_txt}</td>'
                                 f'<td>{t:.6g}</td><td>{g}</td>'
                                 f'<td>{k:.4g}</td></tr>')
                parts.append('</table>')
            except Exception:
                pass

            parts.append('</body></html>')
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(parts))
            self.status_label.setText(f'Report written to {path}')
        except Exception as e:
            QMessageBox.warning(
                self, 'Report failed',
                f'{type(e).__name__}: {e}')

    def _build_thorlabs_menu(self, parent_menu):
        """Group the Thorlabs catalog by family + EFL for findability."""
        from ..prescriptions import THORLABS_CATALOG

        try:
            from .. import raytrace as _rt
        except Exception:
            _rt = None

        def _efl_of(part):
            entry = THORLABS_CATALOG[part]
            kind = entry.get('type', '')
            # Quick paraxial EFL estimate from glass + radii so we can
            # sort within a family.  Falls back to part-name sort if
            # the catalog entry doesn't carry simple-singlet keys.
            try:
                if kind == 'singlet':
                    R1 = entry['R1']
                    R2 = entry['R2']
                    n = 1.5168     # N-BK7 visible -- close enough to sort
                    if 'glass' in entry and entry['glass'] in ('S-LAH64', 'N-SF11'):
                        n = 1.785
                    inv = (n - 1) * (1.0 / R1 - 1.0 / R2)
                    return 1.0 / inv if inv else float('inf')
                if kind == 'doublet':
                    # Use lensmaker on outer surfaces, average glass.
                    R1, R3 = entry['R1'], entry['R3']
                    n = 1.6
                    inv = (n - 1) * (1.0 / R1 - 1.0 / R3)
                    return 1.0 / inv if inv else float('inf')
            except Exception:
                pass
            return float('inf')

        # Bucket by family prefix (LA, LB, AC etc.)
        families = {}
        for part in THORLABS_CATALOG:
            fam = ''.join(c for c in part.split('-')[0]
                          if c.isalpha()) or 'Other'
            families.setdefault(fam, []).append(part)

        FAMILY_LABELS = {
            'LA':  'Plano-convex (LA-)',
            'LB':  'Bi-convex (LB-)',
            'LC':  'Plano-concave (LC-)',
            'LD':  'Bi-concave (LD-)',
            'LE':  'Meniscus (LE-)',
            'LF':  'Aspheric (LF-)',
            'AC':  'Achromatic doublets (AC-)',
            'ACA': 'Achromatic apochromats (ACA-)',
            'ACT': 'Achromatic triplets (ACT-)',
        }

        for fam in sorted(families):
            label = FAMILY_LABELS.get(fam, f'{fam}-')
            sub = parent_menu.addMenu(label)
            # Sort by EFL ascending then part name.
            parts = sorted(families[fam], key=lambda p: (_efl_of(p), p))
            for part in parts:
                f_mm = _efl_of(part) * 1e3
                if f_mm == float('inf') or f_mm != f_mm:
                    text = part
                else:
                    text = f'{part}    (f \u2248 {f_mm:.0f} mm)'
                sub.addAction(text,
                              lambda p=part: self._ins_thorlabs(p))

    def _install_shortcuts(self):
        """Window-level shortcuts that don't already live on a menu action."""
        from PySide6.QtGui import QShortcut

        # Alt+\u2191 / Alt+\u2193: move element up/down in the table.
        QShortcut(QKeySequence('Alt+Up'), self,
                  activated=self.element_editor._move_up)
        QShortcut(QKeySequence('Alt+Down'), self,
                  activated=self.element_editor._move_down)
        # Ctrl+E focuses the element table.
        QShortcut(QKeySequence('Ctrl+E'), self,
                  activated=lambda: self.element_editor.table.setFocus())
        # Shift+Up / Shift+Down: nudge the selected element's distance
        # by +/-0.1 mm without needing to click into the cell first.
        QShortcut(QKeySequence('Shift+Up'), self,
                  activated=lambda: self._nudge_distance(+0.1))
        QShortcut(QKeySequence('Shift+Down'), self,
                  activated=lambda: self._nudge_distance(-0.1))
        # Ctrl+Shift+Up / Ctrl+Shift+Down: bigger nudge (+/-1 mm).
        QShortcut(QKeySequence('Ctrl+Shift+Up'), self,
                  activated=lambda: self._nudge_distance(+1.0))
        QShortcut(QKeySequence('Ctrl+Shift+Down'), self,
                  activated=lambda: self._nudge_distance(-1.0))

        # Ctrl+1 .. Ctrl+9: jump directly to workspace tab N.
        for i in range(1, 10):
            QShortcut(QKeySequence(f'Ctrl+{i}'), self,
                      activated=lambda i=i: self._switch_to_workspace(i - 1))

    def _nudge_distance(self, delta_mm: float):
        """Bump the currently-selected element's axial distance.

        This is a pragmatic keyboard nudge that works from anywhere in
        the window -- the user doesn't have to click into the Distance
        cell first.  Undo-safe (goes through the model's _checkpoint).
        """
        try:
            idx = self.element_editor.table.currentIndex().row()
            if idx <= 0 or idx >= len(self.model.elements):
                return
            self.model._checkpoint()
            self.model.elements[idx].distance_mm = max(
                0.0, self.model.elements[idx].distance_mm + delta_mm)
            self.model._invalidate()
            self.model.system_changed.emit()
            self.status_label.setText(
                f'Nudged E{idx} distance by {delta_mm:+.2f} mm '
                f'(now {self.model.elements[idx].distance_mm:.3f} mm)')
        except Exception as e:
            try:
                from .diagnostics import diag
                diag.report('nudge-distance', e)
            except Exception:
                pass

    def _build_toolbar(self):
        tb = QToolBar('Main')
        tb.setObjectName('main_toolbar')
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addAction('New', self._new_system).setToolTip('Start a fresh system (Ctrl+N)')
        tb.addAction('Open', self._load_file).setToolTip('Open a prescription file (Ctrl+O)')
        tb.addAction('Save', self._save_json_design).setToolTip(
            'Save current design as JSON (Ctrl+S)')

        tb.addSeparator()
        self.tb_undo = tb.addAction('\u21ba Undo')
        self.tb_undo.setToolTip('Undo last edit (Ctrl+Z)')
        self.tb_undo.setEnabled(False)
        self.tb_undo.triggered.connect(self.model.undo)
        self.tb_redo = tb.addAction('\u21bb Redo')
        self.tb_redo.setToolTip('Redo (Ctrl+Y)')
        self.tb_redo.setEnabled(False)
        self.tb_redo.triggered.connect(self.model.redo)

        tb.addSeparator()
        tb.addAction('Insert Lens', self._ins_plano_convex).setToolTip(
            'Insert a plano-convex singlet (Ctrl+L)')
        tb.addAction('Insert Mirror', self._ins_flat_mirror).setToolTip(
            'Insert a flat mirror (Ctrl+M)')

        tb.addSeparator()
        tb.addAction('Retrace', lambda: self.model.run_trace()).setToolTip(
            'Rerun the geometric trace (Ctrl+T)')
        tb.addAction('Run Wave Optics', self._run_waveoptics_now).setToolTip(
            'Kick off a full wave-optics simulation (F5)')
        tb.addAction('Optimize', self._run_local_optimize_toolbar).setToolTip(
            'Run a local geometric optimization with current variables')

        tb.addSeparator()
        tb.addAction('Fit View', self.layout_view.rebuild).setToolTip(
            'Re-fit the 2D layout to the system extent')

    def _on_history_changed(self, can_undo, can_redo):
        if hasattr(self, 'act_undo'):
            self.act_undo.setEnabled(can_undo)
        if hasattr(self, 'act_redo'):
            self.act_redo.setEnabled(can_redo)
        if hasattr(self, 'tb_undo'):
            self.tb_undo.setEnabled(can_undo)
        if hasattr(self, 'tb_redo'):
            self.tb_redo.setEnabled(can_redo)

    def _show_and_raise(self, dock):
        """Make a dock visible + front-most (for diagnostics click, etc.)."""
        dock.show()
        dock.raise_()

    def _run_waveoptics_now(self):
        self._show_and_raise(self.waveoptics_dock)
        if hasattr(self.waveoptics_widget, '_on_run'):
            try:
                self.waveoptics_widget._on_run()
            except Exception as e:
                diag.report('waveoptics-launch', e)

    def _run_local_optimize_toolbar(self):
        self._show_and_raise(self.optimizer_dock)
        if hasattr(self.optimizer_widget, '_start_optimize'):
            try:
                self.optimizer_widget._start_optimize()
            except Exception as e:
                diag.report('optimizer-launch', e)

    def _save_snapshot(self):
        """Ctrl+B: save a named snapshot."""
        name, ok = QInputDialog.getText(
            self, 'Save Snapshot', 'Snapshot name:',
            text=f'Snapshot {len(self.model.snapshots) + 1}')
        if ok and name:
            self.model.save_snapshot(name)
            self._show_and_raise(self.snapshots_dock)

    def _repeat_last_insert(self):
        if self._last_insert_action is not None:
            self._last_insert_action()

    def _show_shortcuts(self):
        lines = [
            'File / Edit',
            '  Ctrl+N         New system',
            '  Ctrl+O         Open prescription (.zmx / .txt)',
            '  Ctrl+S         Save as JSON',
            '  Ctrl+Z         Undo',
            '  Ctrl+Y         Redo',
            '  Ctrl+B         Save snapshot',
            '',
            'Insert',
            '  Ctrl+L         Plano-convex singlet',
            '  Ctrl+Shift+L   Achromatic doublet',
            '  Ctrl+M         Flat mirror',
            '  Ctrl+R         Repeat last insert',
            '  Ctrl+D         Delete selected element',
            '',
            'Table navigation',
            '  Ctrl+E         Focus element table',
            '  Alt+\u2191 / Alt+\u2193  Move selected element up / down',
            '',
            'Analysis',
            '  Ctrl+T         Retrace',
            '  F5             Run Wave Optics',
        ]
        QMessageBox.information(self, 'Keyboard Shortcuts',
                                '\n'.join(lines))

    def _restore_session(self):
        if self.model.restore_session():
            QMessageBox.information(self, 'Restored',
                'Last autosaved session restored.')
        else:
            QMessageBox.information(self, 'No session',
                'No autosaved session was found.')

    # ---- JSON design save/load (native format) ------------------------

    def _save_json_design(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Design (JSON)',
            filter='Optical Designer JSON (*.json)')
        if not path:
            return
        if not path.endswith('.json'):
            path += '.json'
        try:
            data = self.model._state_to_jsonable(self.model._capture_state())
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.status_label.setText(f'Saved {os.path.basename(path)}')
            self._set_current_path(path)
        except Exception as e:
            diag.report('json-save', e, context=path)
            QMessageBox.critical(self, 'Save failed', str(e))

    def _open_json_design(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Design (JSON)',
            filter='Optical Designer JSON (*.json);;All files (*)')
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._suppress_dirty_marking = True
            try:
                state = self.model._state_from_jsonable(data)
                self.model._checkpoint()
                self.model._restore_state(state)
            finally:
                self._suppress_dirty_marking = False
            self.status_label.setText(f'Loaded {os.path.basename(path)}')
            self._set_current_path(path)
        except Exception as e:
            diag.report('json-load', e, context=path)
            QMessageBox.critical(self, 'Load failed', str(e))

    # ---- Analysis-menu helpers ----------------------------------------

    def _zernike_from_trace(self):
        """Compute Zernike coefficients from the geometric OPD at exit
        pupil -- no wave-optics simulation needed.  Fast and useful for
        quick aberration inspection while sliders are being dragged."""
        try:
            from ..raytrace import (make_rings, make_grid, trace,
                                     find_paraxial_focus)
            from ..analysis import zernike_decompose
        except Exception as e:
            diag.report('zernike-trace', e)
            return
        surfaces = self.model.build_trace_surfaces()
        if not surfaces:
            return
        wv = self.model.wavelength_m
        semi_ap = self.model.epd_m / 2
        try:
            rays = make_grid(semi_ap, 25, wv)
            bfl = find_paraxial_focus(surfaces, wv)
            if np.isfinite(bfl):
                from ..raytrace import Surface
                surfaces[-1].thickness = bfl
                surfaces.append(Surface(
                    radius=np.inf, semi_diameter=np.inf,
                    glass_before=surfaces[-1].glass_after,
                    glass_after=surfaces[-1].glass_after))
            result = trace(rays, surfaces, wv)
            # OPD at the image plane = opd - mean(opd), sampled where alive.
            r = result.image_rays
            alive = r.alive
            opd = r.opd[alive] - r.opd[alive].mean()
            x = rays.x[alive]
            y = rays.y[alive]
            # Build a sparse OPD map on a regular grid for zernike_decompose.
            N = 128
            dx = 2 * semi_ap / N
            grid = np.full((N, N), np.nan)
            ix = np.clip(((x / dx) + N / 2).astype(int), 0, N - 1)
            iy = np.clip(((y / dx) + N / 2).astype(int), 0, N - 1)
            grid[iy, ix] = opd
            coeffs, names = zernike_decompose(
                grid, dx, aperture=2 * semi_ap, n_modes=21)
            msg = 'Zernike coefficients from ray-traced OPD (nm RMS):\n'
            for j, (c, name) in enumerate(zip(coeffs, names)):
                if abs(c) > 1e-12:
                    msg += f'  Z{j:2d} {name:30s}{c*1e9:+8.2f} nm\n'
            QMessageBox.information(self, 'Zernikes from trace', msg)
        except Exception as e:
            diag.report('zernike-trace', e)
            QMessageBox.warning(self, 'Zernike decomposition failed', str(e))

    def _run_through_focus(self):
        """Legacy entry point kept for any external callers -- the menu
        now points at the dedicated Through-focus dock directly."""
        self._show_and_raise(self.through_focus_dock)

    def _find_nearest_thorlabs(self):
        """Rank Thorlabs catalog parts by |dEFL| to the current system
        and offer to replace the first lens element with the best match.
        """
        try:
            from ..prescriptions import THORLABS_CATALOG
            from ..raytrace import system_abcd, surfaces_from_prescription
            pres = self.model.to_prescription()
            surfs = surfaces_from_prescription(pres)
            _, target_efl, _, _ = system_abcd(surfs,
                                               self.model.wavelength_nm * 1e-9)
        except Exception as e:
            QMessageBox.warning(self, 'Nearest Thorlabs', str(e))
            return
        from math import isfinite

        def _efl_of_part(name):
            entry = THORLABS_CATALOG[name]
            try:
                if entry.get('type') == 'singlet':
                    R1 = entry['R1']; R2 = entry['R2']
                    n = 1.5168
                    inv = (n - 1) * (1.0 / R1 - 1.0 / R2)
                    return 1.0 / inv if inv else float('inf')
                if entry.get('type') == 'doublet':
                    R1 = entry['R1']; R3 = entry['R3']
                    n = 1.6
                    inv = (n - 1) * (1.0 / R1 - 1.0 / R3)
                    return 1.0 / inv if inv else float('inf')
            except Exception:
                pass
            return float('inf')

        if not isfinite(target_efl):
            QMessageBox.information(
                self, 'Nearest Thorlabs',
                'Current system has no finite EFL to match.')
            return
        ranked = sorted(
            THORLABS_CATALOG.keys(),
            key=lambda p: abs(_efl_of_part(p) - target_efl))
        lines = [
            f'Target EFL = {target_efl*1e3:.3f} mm\n',
            f'{"rank":>5s}  {"part":<14s}  {"EFL (mm)":>10s}  {"dEFL (mm)":>10s}',
        ]
        for i, p in enumerate(ranked[:20]):
            f_p = _efl_of_part(p) * 1e3
            lines.append(f'{i+1:>5d}  {p:<14s}  {f_p:>10.3f}  '
                         f'{f_p - target_efl*1e3:+10.3f}')
        QMessageBox.information(
            self, 'Nearest Thorlabs parts', '\n'.join(lines))

    def _run_chromatic(self):
        """Quick chromatic focal shift using the current wavelengths list."""
        try:
            from ..analysis import chromatic_focal_shift
            from ..raytrace import surfaces_from_prescription, system_abcd
            pres = self.model.to_prescription()
            surfaces = surfaces_from_prescription(pres)
            wls = sorted([w * 1e-9 for w in self.model.wavelengths_nm])
            if len(wls) < 2:
                QMessageBox.information(self, 'Chromatic shift',
                    'Add at least two wavelengths (Optimizer dock \u2192 +\u03bb) '
                    'to compute chromatic focal shift.')
                return
            efls = []
            bfls = []
            for w in wls:
                _, efl, bfl, _ = system_abcd(surfaces, w)
                efls.append(efl * 1e3)
                bfls.append(bfl * 1e3)
            lines = ['Chromatic focal shift:',
                     f'  {"wv(nm)":>10s}  {"EFL(mm)":>10s}  {"BFL(mm)":>10s}']
            for w, ef, bf in zip(wls, efls, bfls):
                lines.append(f'  {w*1e9:10.1f}  {ef:10.3f}  {bf:10.3f}')
            lines.append(f'  EFL P-V: {max(efls) - min(efls):.4f} mm')
            QMessageBox.information(self, 'Chromatic focal shift',
                                    '\n'.join(lines))
        except Exception as e:
            diag.report('chromatic', e)
            QMessageBox.warning(self, 'Failed', str(e))

    def _export_spot_png(self):
        """Save whatever the spot-diagram widget is currently showing."""
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save spot diagram as PNG',
            filter='PNG image (*.png)')
        if not path:
            return
        if not path.endswith('.png'):
            path += '.png'
        try:
            pm = self.spot_widget.grab()
            pm.save(path, 'PNG')
            self.status_label.setText(
                f'Saved {os.path.basename(path)}')
        except Exception as e:
            diag.report('spot-png', e, context=path)

    def _export_python_script(self):
        """Use the codegen module to emit a self-contained sim script."""
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Python sim script',
            filter='Python source (*.py)')
        if not path:
            return
        if not path.endswith('.py'):
            path += '.py'
        try:
            from .. import generate_simulation_script
            rx = self.model.to_prescription()
            code = generate_simulation_script(
                rx, wavelength=self.model.wavelength_m)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(code)
            self.status_label.setText(f'Wrote {os.path.basename(path)}')
        except Exception as e:
            diag.report('codegen', e, context=path)
            QMessageBox.critical(self, 'Export failed', str(e))

    # ── Actions ─────────────────────────────────────────────────────

    def _new_system(self):
        self._suppress_dirty_marking = True
        try:
            self.model.__init__()
            self.model.system_changed.emit()
        finally:
            self._suppress_dirty_marking = False
        self._current_path = None
        self._dirty = False
        self._refresh_window_title()

    def _delete_element(self):
        self.element_editor._delete_element()

    # ── Element dialog helper ──────────────────────────────────────

    def _elem_dialog(self, title, fields, advanced=None):
        """Show a dialog. fields: {key: (label, default, min, max, decimals, suffix)}.

        If ``advanced`` is provided (same dict shape) it is shown behind
        an "Advanced..." expander so quick inserts stay quick but the
        user can override auto-computed defaults when they need to.
        """
        from PySide6.QtWidgets import (
            QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox,
            QSpinBox, QPushButton, QWidget,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(360)
        form = QFormLayout(dlg)

        def add_fields(target_form, field_map):
            out = {}
            for key, (label, val, lo, hi, dec, sfx) in field_map.items():
                if isinstance(val, int) and dec == 0:
                    sp = QSpinBox()
                    sp.setRange(int(lo), int(hi)); sp.setValue(val)
                else:
                    sp = QDoubleSpinBox()
                    sp.setRange(lo, hi); sp.setValue(val); sp.setDecimals(dec)
                if sfx:
                    sp.setSuffix(f' {sfx}')
                target_form.addRow(f'{label}:', sp)
                out[key] = sp
            return out

        widgets = add_fields(form, fields)

        # --- Advanced expander ---
        adv_widgets = {}
        if advanced:
            adv_panel = QWidget()
            adv_form = QFormLayout(adv_panel)
            adv_form.setContentsMargins(0, 0, 0, 0)
            adv_widgets = add_fields(adv_form, advanced)
            adv_panel.hide()
            btn_adv = QPushButton('Advanced \u25bc')
            btn_adv.setCheckable(True)

            def toggle_advanced():
                if btn_adv.isChecked():
                    btn_adv.setText('Advanced \u25b2')
                    adv_panel.show()
                else:
                    btn_adv.setText('Advanced \u25bc')
                    adv_panel.hide()
            btn_adv.clicked.connect(toggle_advanced)
            form.addRow(btn_adv)
            form.addRow(adv_panel)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        if dlg.exec() != QDialog.Accepted:
            return {}, False
        out = {k: w.value() for k, w in widgets.items()}
        out.update({k: w.value() for k, w in adv_widgets.items()})
        return out, True

    # ── Insert elements ────────────────────────────────────────────

    def _ins_before_detector(self, elem):
        self.model.insert_element(len(self.model.elements) - 1, elem)

    def _ins_plano_convex(self):
        f_default = 100.0
        p, ok = self._elem_dialog(
            'Insert Plano-Convex Singlet',
            {
                'f':    ('Focal length',           f_default, 1, 10000, 1, 'mm'),
                'dist': ('Distance from previous', 50.0,      0, 1e6,   2, 'mm'),
            },
            advanced={
                't':    ('Center thickness (0 = auto)', 0.0,  0, 100,   3, 'mm'),
                'sd':   ('Semi-diameter (0 = auto)',    0.0,  0, 100,   3, 'mm'),
                'n':    ('Glass index (approx)',        1.517, 1.3, 2.2, 4, ''),
            },
        )
        if not ok: return
        f, d = p['f'], p['dist']
        n = p.get('n', 1.517)
        R = f * (n - 1)
        t = p.get('t') or max(f * 0.04, 2)
        sd = p.get('sd') or f * 0.12
        self._ins_before_detector(Element(0, f'PC f={f:.0f}', 'Singlet', d, surfaces=[
            SurfaceRow(R, t, 'N-BK7', sd), SurfaceRow(np.inf, 0, '', sd)]))
        self._last_insert_action = self._ins_plano_convex

    def _ins_biconvex(self):
        p, ok = self._elem_dialog(
            'Insert Biconvex Singlet',
            {
                'f':    ('Focal length', 100, 1, 10000, 1, 'mm'),
                'dist': ('Distance from previous', 50, 0, 1e6, 2, 'mm'),
            },
            advanced={
                't':  ('Center thickness (0 = auto)',   0.0, 0, 100, 3, 'mm'),
                'sd': ('Semi-diameter (0 = auto)',      0.0, 0, 100, 3, 'mm'),
                'n':  ('Glass index (approx)',          1.517, 1.3, 2.2, 4, ''),
            })
        if not ok: return
        f, d = p['f'], p['dist']
        n = p.get('n', 1.517)
        R = 2 * f * (n - 1)   # biconvex: 1/f = (n-1)*(1/R1 - 1/R2) with R1=-R2
        t = p.get('t') or max(f * 0.05, 2)
        sd = p.get('sd') or f * 0.12
        self._ins_before_detector(Element(0, f'BX f={f:.0f}', 'Singlet', d, surfaces=[
            SurfaceRow(R, t, 'N-BK7', sd), SurfaceRow(-R, 0, '', sd)]))
        self._last_insert_action = self._ins_biconvex

    def _ins_doublet(self):
        p, ok = self._elem_dialog('Insert Achromatic Doublet', {
            'f': ('Focal length', 100, 1, 10000, 1, 'mm'),
            'dist': ('Distance from previous', 50, 0, 1e6, 2, 'mm'),
        })
        if not ok: return
        f, d = p['f'], p['dist']; s = f / 100; sd = 12.7 * s
        self._ins_before_detector(Element(0, f'Doublet f={f:.0f}', 'Doublet', d, surfaces=[
            SurfaceRow(62.8*s, 6*s, 'N-BAF10', sd),
            SurfaceRow(-46.5*s, 2.5*s, 'N-SF6HT', sd),
            SurfaceRow(-184.5*s, 0, '', sd)]))
        self._last_insert_action = self._ins_doublet

    def _ins_cylindrical(self):
        p, ok = self._elem_dialog('Insert Cylindrical Lens', {
            'R': ('Radius of curvature', 50, 1, 10000, 2, 'mm'),
            't': ('Center thickness', 3, 0.1, 50, 2, 'mm'),
            'dist': ('Distance from previous', 50, 0, 1e6, 2, 'mm'),
        })
        if not ok:
            return
        # Ask which axis carries the curvature.  X-axis (the default
        # in 2.5) means the lens focuses in X; Y-axis swaps the radii.
        axis, ok2 = QInputDialog.getItem(
            self, 'Cylindrical axis',
            'Which axis carries the curvature?',
            ['X (focuses in X)', 'Y (focuses in Y)'], 0, False)
        if not ok2:
            return
        R, t, d = p['R'], p['t'], p['dist']
        sd = self.model.epd_mm / 2
        if axis.startswith('X'):
            r_x, r_y = R, np.inf
            label = f'Cyl-X R={R:.0f}'
        else:
            r_x, r_y = np.inf, R
            label = f'Cyl-Y R={R:.0f}'
        self._ins_before_detector(Element(
            0, label, 'Singlet', d, surfaces=[
                SurfaceRow(r_x, t, 'N-BK7', sd,
                           radius_y=r_y, conic_y=0.0),
                SurfaceRow(np.inf, 0, '', sd,
                           radius_y=np.inf, conic_y=0.0)]))

    def _ins_biconic(self):
        p, ok = self._elem_dialog('Insert Biconic Singlet', {
            'Rx': ('Radius X', 50, 1, 10000, 2, 'mm'),
            'Ry': ('Radius Y', 70, 1, 10000, 2, 'mm'),
            't': ('Center thickness', 4, 0.1, 50, 2, 'mm'),
            'dist': ('Distance from previous', 50, 0, 1e6, 2, 'mm'),
        })
        if not ok:
            return
        Rx, Ry, t, d = p['Rx'], p['Ry'], p['t'], p['dist']
        sd = self.model.epd_mm / 2
        self._ins_before_detector(Element(
            0, f'Biconic {Rx:.0f}/{Ry:.0f}', 'Singlet', d, surfaces=[
                SurfaceRow(Rx, t, 'N-BK7', sd, radius_y=Ry, conic_y=0.0),
                SurfaceRow(np.inf, 0, '', sd)]))

    def _ins_flat_mirror(self):
        p, ok = self._elem_dialog('Insert Flat Mirror', {
            'dist': ('Distance from previous', 50, 0, 1e6, 2, 'mm'),
        })
        if not ok: return
        sd = self.model.epd_mm / 2
        self._ins_before_detector(Element(0, 'Flat mirror', 'Mirror', p['dist'], surfaces=[
            SurfaceRow(np.inf, 0, '', sd, surf_type='Mirror')]))

    def _ins_curved_mirror(self):
        p, ok = self._elem_dialog('Insert Curved Mirror', {
            'f': ('Focal length', 100, -1e4, 1e4, 1, 'mm'),
            'dist': ('Distance from previous', 50, 0, 1e6, 2, 'mm'),
        })
        if not ok: return
        sd = self.model.epd_mm / 2
        self._ins_before_detector(Element(0, f'Mirror f={p["f"]:.0f}', 'Mirror', p['dist'],
            surfaces=[SurfaceRow(2*p['f'], 0, '', sd, surf_type='Mirror')]))

    def _ins_mla(self):
        p, ok = self._elem_dialog('Insert Microlens Array', {
            'pitch': ('Lenslet pitch', 0.050, 0.001, 10, 4, 'mm'),
            'f': ('Lenslet focal length', 0.210, 0.001, 100, 4, 'mm'),
            'nx': ('Array size NxN', 12, 1, 1000, 0, ''),
            'sd': ('Semi-diameter', 5.0, 0.01, 100, 2, 'mm'),
            'dist': ('Distance from previous', 50, 0, 1e6, 2, 'mm'),
        })
        if not ok: return
        nx = int(p['nx'])
        self._ins_before_detector(Element(0, f'MLA {nx}x{nx}', 'MLA', p['dist'],
            surfaces=[SurfaceRow(np.inf, 0, '', p['sd'])],
            aux={'pitch_mm': p['pitch'], 'f_mm': p['f'], 'nx': nx, 'ny': nx}))

    def _ins_grating(self):
        p, ok = self._elem_dialog('Insert Diffraction Grating', {
            'period': ('Grating period', 0.010, 1e-4, 10, 4, 'mm'),
            'sd': ('Semi-diameter', 5.0, 0.01, 100, 2, 'mm'),
            'dist': ('Distance from previous', 50, 0, 1e6, 2, 'mm'),
        })
        if not ok: return
        self._ins_before_detector(Element(0, f'Grating p={p["period"]:.4f}', 'DOE', p['dist'],
            surfaces=[SurfaceRow(np.inf, 0, '', p['sd'])],
            aux={'period_mm': p['period'], 'type': 'grating'}))

    def _ins_dammann(self):
        p, ok = self._elem_dialog('Insert Dammann Grating', {
            'period': ('Period', 0.164, 0.001, 10, 4, 'mm'),
            'orders': ('Orders NxN', 12, 2, 64, 0, ''),
            'levels': ('Phase levels', 8, 2, 32, 0, ''),
            'sd': ('Semi-diameter', 5.0, 0.01, 100, 2, 'mm'),
            'dist': ('Distance from prev', 50, 0, 1e6, 2, 'mm'),
        })
        if not ok: return
        n = int(p['orders'])
        self._ins_before_detector(Element(0, f'Dammann {n}x{n}', 'Dammann', p['dist'],
            surfaces=[SurfaceRow(np.inf, 0, '', p['sd'])],
            aux={'period_mm': p['period'], 'orders_x': n, 'orders_y': n,
                 'phase_levels': int(p['levels'])}))

    def _ins_thorlabs(self, part):
        from ..prescriptions import thorlabs_lens
        try:
            p, ok = self._elem_dialog(f'Insert {part}', {
                'dist': ('Distance from previous', 50, 0, 1e6, 2, 'mm'),
            })
            if not ok: return
            rx = thorlabs_lens(part)
            surfs = rx.get('surfaces', []); thick = rx.get('thicknesses', [])
            ap = rx.get('aperture_diameter'); sd = ap * 1e3 / 2 if ap else np.inf
            rows = []
            for i, ps in enumerate(surfs):
                R = ps['radius'] * 1e3 if np.isfinite(ps['radius']) else np.inf
                t = thick[i] * 1e3 if i < len(thick) else 0.0
                g = ps.get('glass_after', ''); g = '' if g == 'air' else g
                rows.append(SurfaceRow(R, t, g, sd, ps.get('conic', 0.0)))
            et = 'Doublet' if len(rows) == 3 else 'Singlet'
            self._ins_before_detector(Element(0, part, et, p['dist'], surfaces=rows))
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    # ── Color preferences ──────────────────────────────────────────

    def _pick_color(self, pref_key, title):
        from PySide6.QtWidgets import QColorDialog
        current = QColor(self.model.prefs.get(pref_key, '#ffffff'))
        color = QColorDialog.getColor(current, self, title)
        if color.isValid():
            self.model.prefs[pref_key] = color.name()
            self.model.display_changed.emit()
            self.status_label.setText(f'{title}: {color.name()}')

    def _pick_bg_2d(self):
        self._pick_color('bg_2d', '2D Layout Background')
        # Apply immediately
        self.layout_view.scene.setBackgroundBrush(
            QColor(self.model.prefs['bg_2d']))

    def _pick_bg_3d(self):
        self._pick_color('bg_3d', '3D Layout Background')
        if self.layout3d_view._plotter:
            self.layout3d_view._plotter.set_background(
                self.model.prefs['bg_3d'])

    def _pick_ray_color(self):
        self._pick_color('ray_color', 'Ray Color')
        self.model.prefs['ray_use_wavelength'] = False
        self.model.run_trace()  # retrace to update ray display

    def _toggle_wavelength_colors(self):
        self.model.prefs['ray_use_wavelength'] = not self.model.prefs['ray_use_wavelength']
        self.model.run_trace()

    def _pick_accent(self):
        self._pick_color('accent', 'UI Highlight Color')
        # Re-apply theme with the new accent
        apply_theme(QApplication.instance(), self.model.prefs)

    def _set_theme(self, theme_name):
        self.model.prefs['theme'] = theme_name
        apply_theme(QApplication.instance(), self.model.prefs)
        self.status_label.setText(f'Theme: {theme_name}')

    # ── File operations ────────────────────────────────────────────

    def _load_thorlabs(self):
        from ..prescriptions import THORLABS_CATALOG, thorlabs_lens
        parts = sorted(THORLABS_CATALOG.keys())
        part, ok = QInputDialog.getItem(self, 'Load Thorlabs Lens', 'Part:', parts, 0, False)
        if ok and part:
            try:
                rx = thorlabs_lens(part)
                self.model.load_prescription(rx, self.model.wavelength_nm)
                self.status_label.setText(f'Loaded {part}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))

    def _load_file(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, 'Open Prescription File', '',
            'Prescription files (*.zmx *.txt *.seq);;'
            'Zemax (*.zmx *.txt);;CODE V sequence (*.seq);;All files (*)')
        if not fp: return
        try:
            lower = fp.lower()
            self._suppress_dirty_marking = True
            try:
                if lower.endswith('.zmx'):
                    from ..prescriptions import load_zmx_prescription
                    rx = load_zmx_prescription(fp)
                elif lower.endswith('.seq'):
                    from ..prescriptions import load_codev_seq
                    rx = load_codev_seq(fp)
                else:
                    from ..prescriptions import load_zemax_prescription_txt
                    rx = load_zemax_prescription_txt(fp)
                wv = rx.get('wavelength', self.model.wavelength_nm * 1e-9)
                if isinstance(wv, float) and wv < 1e-3:
                    wv *= 1e9
                self.model.load_prescription(rx, wv)
            finally:
                self._suppress_dirty_marking = False
            self.status_label.setText(f'Loaded {rx.get("name", fp)}')
            self._set_current_path(fp)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def _export_prescription(self):
        fp, _ = QFileDialog.getSaveFileName(
            self, 'Export', 'prescription.json',
            'JSON (*.json);;Zemax ZMX (*.zmx);;CODE V sequence (*.seq)')
        if not fp: return
        rx = self.model.to_prescription()
        lower = fp.lower()
        try:
            if lower.endswith('.zmx'):
                from ..prescriptions import export_zemax_zmx
                export_zemax_zmx(rx, fp,
                                 wavelength=self.model.wavelength_nm * 1e-9)
            elif lower.endswith('.seq'):
                from ..prescriptions import export_codev_seq
                export_codev_seq(rx, fp,
                                 wavelength=self.model.wavelength_nm * 1e-9)
            else:
                def conv(o):
                    if isinstance(o, (np.integer,)): return int(o)
                    if isinstance(o, (np.floating,)): return float(o)
                    if isinstance(o, np.ndarray): return o.tolist()
                    return o
                with open(fp, 'w') as f:
                    json.dump(rx, f, indent=2, default=conv)
            self.status_label.setText(f'Exported to {fp}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def _update_status(self):
        n = self.model.num_elements - 2
        efl = self.model.efl_mm
        bfl = self.model.bfl_mm
        epd = self.model.epd_mm
        wv  = self.model.wavelength_nm
        efl_s = f'{efl:.2f}' if np.isfinite(efl) else '—'
        bfl_s = f'{bfl:.2f}' if np.isfinite(bfl) else '—'
        if np.isfinite(efl) and epd > 0:
            fnum_s = f'{abs(efl) / epd:.2f}'
        else:
            fnum_s = '—'
        self.status_label.setText(f'{n} elements')
        # Permanent right-side metrics
        self.metric_efl.setText(f'EFL {efl_s} mm')
        self.metric_bfl.setText(f'BFL {bfl_s} mm')
        self.metric_fnum.setText(f'f/# {fnum_s}')
        self.metric_epd.setText(f'EPD {epd:.3g} mm')
        self.metric_wv.setText(f'λ {wv:.1f} nm')

    def _mark_dirty(self):
        """Flip dirty=True and refresh the title.  Suppressed during
        programmatic state restores so loads don't immediately show '*'."""
        if self._suppress_dirty_marking:
            return
        if not self._dirty:
            self._dirty = True
            self._refresh_window_title()

    def _set_current_path(self, path):
        self._current_path = path
        self._dirty = False
        self._refresh_window_title()
        if path:
            self._push_recent_file(path)

    _RECENT_LIMIT = 10

    def _recent_files(self):
        try:
            from PySide6.QtCore import QSettings
            s = QSettings('lumenairy', 'OpticalDesigner')
            v = s.value('recent_files') or []
            if isinstance(v, str):
                v = [v]
            return [p for p in v if p]
        except Exception:
            return []

    def _push_recent_file(self, path):
        try:
            from PySide6.QtCore import QSettings
            recents = [p for p in self._recent_files() if p != path]
            recents.insert(0, path)
            recents = recents[:self._RECENT_LIMIT]
            s = QSettings('lumenairy', 'OpticalDesigner')
            s.setValue('recent_files', recents)
            if hasattr(self, 'welcome_widget'):
                self.welcome_widget.set_recent_files(recents)
        except Exception:
            pass

    def _open_recent(self, path):
        """Open a path clicked in the Welcome dock recent-files list."""
        if not path or not os.path.exists(path):
            QMessageBox.warning(
                self, 'Open Recent', f'File not found:\n{path}')
            return
        lower = path.lower()
        try:
            self._suppress_dirty_marking = True
            try:
                if lower.endswith('.json'):
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    state = self.model._state_from_jsonable(data)
                    self.model._checkpoint()
                    self.model._restore_state(state)
                else:
                    if lower.endswith('.zmx'):
                        from ..prescriptions import load_zmx_prescription
                        rx = load_zmx_prescription(path)
                    elif lower.endswith('.seq'):
                        from ..prescriptions import load_codev_seq
                        rx = load_codev_seq(path)
                    else:
                        from ..prescriptions import load_zemax_prescription_txt
                        rx = load_zemax_prescription_txt(path)
                    wv = rx.get('wavelength', self.model.wavelength_nm * 1e-9)
                    if isinstance(wv, float) and wv < 1e-3:
                        wv *= 1e9
                    self.model.load_prescription(rx, wv)
            finally:
                self._suppress_dirty_marking = False
            self._set_current_path(path)
            self.status_label.setText(f'Loaded {os.path.basename(path)}')
        except Exception as e:
            diag.report('open-recent', e, context=path)
            QMessageBox.critical(self, 'Load failed', str(e))

    def _open_demo(self):
        """Load the AC254-100-C demo lens (wired to the Welcome dock)."""
        try:
            from ..prescriptions import thorlabs_lens
            self._suppress_dirty_marking = True
            try:
                self.model.load_prescription(thorlabs_lens('AC254-100-C'), 1310.0)
            finally:
                self._suppress_dirty_marking = False
            self._dirty = False
            self._refresh_window_title()
            self.status_label.setText('Loaded demo: AC254-100-C')
        except Exception as e:
            QMessageBox.critical(self, 'Demo failed', str(e))

    def _refresh_window_title(self):
        base = 'Optical Designer'
        if self._current_path:
            name = os.path.basename(self._current_path)
            star = '*' if self._dirty else ''
            self.setWindowTitle(f'{base} — {name}{star}')
        else:
            star = '*' if self._dirty else ''
            self.setWindowTitle(f'{base}{(" — Untitled" + star) if star else ""}')

    def _switch_to_workspace(self, idx):
        """Programmatically switch to workspace at idx (Ctrl+digit)."""
        try:
            mgr = self.workspace_mgr
        except AttributeError:
            return
        if not (0 <= idx < len(mgr.workspaces)):
            return
        if idx == mgr.current_index:
            return
        mgr.save_current_layout()
        mgr.apply_index(idx)
        self.workspace_bar.set_current_index(idx)

    # ---- Drag-and-drop file open ------------------------------------

    _DROP_EXTS = ('.zmx', '.txt', '.seq', '.json')

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            for url in md.urls():
                fp = url.toLocalFile()
                if fp.lower().endswith(self._DROP_EXTS):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        md = event.mimeData()
        if not md.hasUrls():
            event.ignore()
            return
        for url in md.urls():
            fp = url.toLocalFile()
            lower = fp.lower()
            if not lower.endswith(self._DROP_EXTS):
                continue
            try:
                self._suppress_dirty_marking = True
                if lower.endswith('.json'):
                    with open(fp, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    state = self.model._state_from_jsonable(data)
                    self.model._checkpoint()
                    self.model._restore_state(state)
                else:
                    if lower.endswith('.zmx'):
                        from ..prescriptions import load_zmx_prescription
                        rx = load_zmx_prescription(fp)
                    elif lower.endswith('.seq'):
                        from ..prescriptions import load_codev_seq
                        rx = load_codev_seq(fp)
                    else:
                        from ..prescriptions import load_zemax_prescription_txt
                        rx = load_zemax_prescription_txt(fp)
                    wv = rx.get('wavelength', self.model.wavelength_nm * 1e-9)
                    if isinstance(wv, float) and wv < 1e-3:
                        wv *= 1e9
                    self.model.load_prescription(rx, wv)
            except Exception as e:
                diag.report('drop-load', e, context=fp)
                QMessageBox.critical(self, 'Load failed', str(e))
                self._suppress_dirty_marking = False
                continue
            finally:
                self._suppress_dirty_marking = False
            self._set_current_path(fp)
            self.status_label.setText(f'Loaded {os.path.basename(fp)}')
            event.acceptProposedAction()
            return  # only handle the first valid file
        event.ignore()

    def _show_about(self):
        QMessageBox.about(self, 'About Optical Designer',
            '<h3>Optical Designer</h3>'
            '<p>Open-source optical design application.</p>'
            f'<p>Library version: {_get_version()}</p>'
            '<p>Author: Andrew Traverso</p>')

    def _open_command_palette(self):
        """Manual entry point for the command palette (Help menu).

        The Ctrl+K / Ctrl+Shift+P shortcut also opens the same dialog
        via the QShortcut installed in __init__.
        """
        from .command_palette import CommandPalette
        dlg = CommandPalette(self, self)
        try:
            geom = self.geometry()
            dlg.move(
                geom.x() + (geom.width() - dlg.width()) // 2,
                geom.y() + 60,
            )
        except Exception:
            pass
        dlg.exec()

    def _show_lens_options(self):
        """Open the tabbed lens-function options dialog.

        Stores the user's choices on ``model.lens_options``; the
        Wave-Optics dock reads them back when delegating to the
        real-lens functions.
        """
        from .lens_options_dialog import LensOptionsDialog
        dlg = LensOptionsDialog(self.model, self)
        if dlg.exec() == 1:  # QDialog.Accepted
            # Refresh the wave-optics forecast so the user can see
            # immediately if their changes matter for cost (e.g.
            # turning off parallel_amp pushes the forecast up).
            try:
                self.waveoptics_widget._update_forecast()
            except Exception:
                pass
            self.status_label.setText('Lens function options updated.')


def _get_version():
    try:
        from .. import __version__
        return __version__
    except Exception:
        return '?'


def _cli_main():
    """Entry point for the optical-designer command."""
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName('Optical Designer')
    apply_dark_theme(app)
    window = MainWindow()
    args = sys.argv[1:]
    if '--demo' in args:
        from ..prescriptions import thorlabs_lens
        window.model.load_prescription(thorlabs_lens('AC254-100-C'), 1310.0)
    elif args and not args[0].startswith('-'):
        fp = args[0]
        try:
            lower = fp.lower()
            if lower.endswith('.zmx'):
                from ..prescriptions import load_zmx_prescription
                rx = load_zmx_prescription(fp)
            elif lower.endswith('.seq'):
                from ..prescriptions import load_codev_seq
                rx = load_codev_seq(fp)
            else:
                from ..prescriptions import load_zemax_prescription_txt
                rx = load_zemax_prescription_txt(fp)
            wv = rx.get('wavelength', 1310e-9)
            if isinstance(wv, float) and wv < 1e-3: wv *= 1e9
            window.model.load_prescription(rx, wv)
        except Exception as e:
            print(f'Error: {e}', file=sys.stderr)
    window.show()
    sys.exit(app.exec())


# ════════════════════════════════════════════════════════════════════════
# Themes
# ════════════════════════════════════════════════════════════════════════

THEMES = {
    'dark': {
        'bg': '#0a0c10', 'panel': '#12161e', 'base': '#0c1018',
        'alt_base': '#0e1118', 'text': '#dde8f8', 'dim': '#7a94b8',
        'border': '#2a3548', 'btn': '#1a2030', 'hover': '#243048',
        'grid': '#1a2030', 'highlight_text': '#ffffff',
    },
    'light': {
        'bg': '#f0f2f5', 'panel': '#ffffff', 'base': '#ffffff',
        'alt_base': '#f5f6f8', 'text': '#1a1a2e', 'dim': '#6a7a8a',
        'border': '#c0c8d4', 'btn': '#e8ecf0', 'hover': '#d0d8e4',
        'grid': '#e0e4e8', 'highlight_text': '#ffffff',
    },
    'midnight': {
        'bg': '#0d1117', 'panel': '#161b22', 'base': '#0d1117',
        'alt_base': '#111820', 'text': '#c9d1d9', 'dim': '#8b949e',
        'border': '#30363d', 'btn': '#21262d', 'hover': '#30363d',
        'grid': '#21262d', 'highlight_text': '#ffffff',
    },
}


def apply_theme(app, prefs=None):
    """Apply a theme with configurable accent color."""
    if prefs is None:
        prefs = {'theme': 'dark', 'accent': '#5cb8ff'}

    theme_name = prefs.get('theme', 'dark')
    accent = prefs.get('accent', '#5cb8ff')
    t = THEMES.get(theme_name, THEMES['dark'])

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(t['bg']))
    pal.setColor(QPalette.WindowText, QColor(t['text']))
    pal.setColor(QPalette.Base, QColor(t['base']))
    pal.setColor(QPalette.AlternateBase, QColor(t['alt_base']))
    pal.setColor(QPalette.Text, QColor(t['text']))
    pal.setColor(QPalette.Button, QColor(t['btn']))
    pal.setColor(QPalette.ButtonText, QColor(t['text']))
    pal.setColor(QPalette.Highlight, QColor(accent))
    pal.setColor(QPalette.HighlightedText, QColor(t['highlight_text']))
    pal.setColor(QPalette.ToolTipBase, QColor(t['panel']))
    pal.setColor(QPalette.ToolTipText, QColor(t['text']))
    pal.setColor(QPalette.PlaceholderText, QColor(t['dim']))
    pal.setColor(QPalette.Mid, QColor(t['border']))
    pal.setColor(QPalette.Dark, QColor(t['bg']))
    pal.setColor(QPalette.Shadow, QColor('#000000'))
    app.setPalette(pal)

    # Build stylesheet with the theme colors and accent
    bg = t['bg']; panel = t['panel']; text = t['text']; dim = t['dim']
    border = t['border']; btn = t['btn']; hover = t['hover']; grid = t['grid']

    app.setStyleSheet(f"""
        QMainWindow {{ background: {bg}; }}
        QMainWindow::separator {{ background: {border}; width: 4px; height: 4px; }}
        QMainWindow::separator:hover {{ background: {accent}; }}
        QDockWidget {{ color: {dim}; font-size: 12px; font-weight: bold; border: 1px solid {border}; }}
        QDockWidget::title {{ background: {panel}; padding: 6px; border-bottom: 1px solid {border}; }}
        QTabBar::tab {{ background: {panel}; color: {dim}; padding: 6px 14px; border: 1px solid {border}; border-bottom: none; font-family: Consolas; font-size: 11px; }}
        QTabBar::tab:selected {{ background: {hover}; color: {text}; border-bottom: 2px solid {accent}; }}
        QTabBar::tab:hover {{ background: {hover}; color: {text}; }}
        QMenuBar {{ background: {panel}; color: {text}; border-bottom: 1px solid {border}; }}
        QMenuBar::item:selected {{ background: {hover}; }}
        QMenu {{ background: {panel}; color: {text}; border: 1px solid {border}; }}
        QMenu::item:selected {{ background: {hover}; }}
        QToolBar {{ background: {panel}; border-bottom: 1px solid {border}; spacing: 4px; padding: 2px; }}
        QToolButton {{ background: {btn}; border: 1px solid {border}; color: {text}; padding: 4px 10px; font-family: Consolas; font-size: 12px; }}
        QToolButton:hover {{ background: {hover}; border-color: {accent}; }}
        QToolButton:pressed {{ background: {hover}; }}
        QStatusBar {{ background: {panel}; color: {dim}; border-top: 1px solid {border}; font-family: Consolas; font-size: 12px; }}
        QTableView {{ background: {bg}; alternate-background-color: {t['alt_base']}; color: {text}; gridline-color: {grid}; border: none; font-family: Consolas; font-size: 12px; selection-background-color: {hover}; }}
        QHeaderView::section {{ background: {panel}; color: {dim}; border: 1px solid {grid}; padding: 4px; font-family: Consolas; font-size: 11px; font-weight: bold; }}
        QLineEdit {{ background: {btn}; color: {text}; border: 1px solid {border}; padding: 3px 6px; font-family: Consolas; font-size: 12px; }}
        QLineEdit:focus {{ border-color: {accent}; }}
        QPushButton {{ background: {btn}; border: 1px solid {border}; color: {text}; padding: 5px 12px; font-family: Consolas; font-size: 12px; }}
        QPushButton:hover {{ background: {hover}; border-color: {accent}; }}
        QPushButton:pressed {{ background: {hover}; }}
        QPushButton:checked {{ background: {accent}; color: {t['highlight_text']}; }}
        QLabel {{ color: {text}; font-family: Consolas; font-size: 12px; }}
        QComboBox {{ background: {btn}; color: {text}; border: 1px solid {border}; padding: 3px 6px; font-family: Consolas; }}
        QComboBox:hover {{ border-color: {accent}; }}
        QComboBox QAbstractItemView {{ background: {panel}; color: {text}; border: 1px solid {border}; selection-background-color: {hover}; }}
        QScrollBar:vertical {{ background: {bg}; width: 10px; border: none; }}
        QScrollBar::handle:vertical {{ background: {border}; border-radius: 4px; min-height: 20px; }}
        QScrollBar::handle:vertical:hover {{ background: {dim}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar:horizontal {{ background: {bg}; height: 10px; border: none; }}
        QScrollBar::handle:horizontal {{ background: {border}; border-radius: 4px; min-width: 20px; }}
        QScrollBar::handle:horizontal:hover {{ background: {dim}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        QSplitter::handle {{ background: {border}; }}
        QSplitter::handle:hover {{ background: {accent}; }}
        QGroupBox {{ border: 1px solid {border}; margin-top: 8px; padding-top: 12px; color: {dim}; font-family: Consolas; font-size: 11px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
        QSpinBox, QDoubleSpinBox {{ background: {btn}; color: {text}; border: 1px solid {border}; padding: 2px 4px; font-family: Consolas; }}
        QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {accent}; }}
        QTextEdit {{ background: {bg}; color: {dim}; border: none; }}
        QProgressBar {{ background: {btn}; border: 1px solid {border}; text-align: center; color: {text}; font-size: 11px; }}
        QProgressBar::chunk {{ background: {accent}; }}
        QCheckBox {{ color: {text}; }}
        QCheckBox::indicator {{ width: 14px; height: 14px; }}
        QCheckBox::indicator:checked {{ background: {accent}; border: 1px solid {accent}; }}
    """)


def apply_dark_theme(app):
    """Backward-compatible entry point — applies dark theme with default accent."""
    apply_theme(app, {'theme': 'dark', 'accent': '#5cb8ff'})

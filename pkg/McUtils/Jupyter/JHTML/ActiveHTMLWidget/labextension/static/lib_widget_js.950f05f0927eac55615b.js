"use strict";
(self["webpackChunkActiveHTMLWidget"] = self["webpackChunkActiveHTMLWidget"] || []).push([["lib_widget_js"],{

/***/ "./lib/version.js":
/*!************************!*\
  !*** ./lib/version.js ***!
  \************************/
/***/ ((__unused_webpack_module, exports, __webpack_require__) => {


// Copyright (c) b3m2a1
// Distributed under the terms of the Modified BSD License.
Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.MODULE_NAME = exports.MODULE_VERSION = void 0;
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
// eslint-disable-next-line @typescript-eslint/no-var-requires
const data = __webpack_require__(/*! ../package.json */ "./package.json");
/**
 * The _model_module_version/_view_module_version this package implements.
 *
 * The html widget manager assumes that this is the same as the npm package
 * version number.
 */
exports.MODULE_VERSION = data.version;
/*
 * The current package name.
 */
exports.MODULE_NAME = data.name;
//# sourceMappingURL=version.js.map

/***/ }),

/***/ "./lib/widget.js":
/*!***********************!*\
  !*** ./lib/widget.js ***!
  \***********************/
/***/ (function(__unused_webpack_module, exports, __webpack_require__) {


// Copyright (c) b3m2a1
// Distributed under the terms of the Modified BSD License.
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.ActiveHTMLView = exports.ActiveHTMLModel = void 0;
const base_1 = __webpack_require__(/*! @jupyter-widgets/base */ "webpack/sharing/consume/default/@jupyter-widgets/base");
const version_1 = __webpack_require__(/*! ./version */ "./lib/version.js");
const widgets_1 = __webpack_require__(/*! @lumino/widgets */ "webpack/sharing/consume/default/@lumino/widgets");
const algorithm_1 = __webpack_require__(/*! @lumino/algorithm */ "webpack/sharing/consume/default/@lumino/algorithm");
const messaging_1 = __webpack_require__(/*! @lumino/messaging */ "webpack/sharing/consume/default/@lumino/messaging");
const jquery_1 = __importStar(__webpack_require__(/*! jquery */ "./node_modules/jquery/dist/jquery.js")), jquery = jquery_1;
const bootstrap = __importStar(__webpack_require__(/*! bootstrap */ "webpack/sharing/consume/default/bootstrap/bootstrap"));
class LayoutManagerWidget extends widgets_1.Widget {
    constructor(options) {
        let view = options.view;
        //@ts-ignore
        options.tag = view.tagName;
        super(options);
        this._view = view;
        this.layout = new widgets_1.PanelLayout({ fitPolicy: 'set-no-constraint' });
    }
    dispose() {
        if (this.isDisposed) {
            return;
        }
        super.dispose();
        if (this._view) {
            this._view.remove();
        }
        //@ts-ignore
        this._view = null;
    }
    processMessage(msg) {
        super.processMessage(msg);
        this._view.processPhosphorMessage(msg);
    }
    get widgets() {
        return this.layout.widgets;
    }
    addWidget(widget) {
        this.layout.addWidget(widget);
    }
    insertWidget(index, widget) {
        this.layout.insertWidget(index, widget);
    }
}
class ActiveHTMLModel extends base_1.DOMWidgetModel {
    // _ihandlers: Record<string, [number, any]>;
    // constructor() {
    //     super();
    // }
    initialize(attributes, options) {
        super.initialize(attributes, options);
        // this._ihandlers= {};
        this._updateHandlers();
        this.on('change:jsHandlers', this._updateHandlers, this);
    }
    defaults() {
        return Object.assign(Object.assign({}, super.defaults()), { _model_name: ActiveHTMLModel.model_name, _model_module: ActiveHTMLModel.model_module, _model_module_version: ActiveHTMLModel.model_module_version, _view_name: ActiveHTMLModel.view_name, _view_module: ActiveHTMLModel.view_module, _view_module_version: ActiveHTMLModel.view_module_version, tagName: 'div', children: [], classList: [], innerHTML: "", textContent: "", _bodyType: "", _debugPrint: false, styleDict: {}, elementAttributes: {}, id: "", value: "", trackInput: false, continuousUpdate: true, eventPropertiesDict: {}, defaultEventProperties: [
                "bubbles", "cancelable", "composed",
                "target", "timestamp", "type",
                "key", "repeat",
                "button", "buttons",
                "alKey", "shiftKey", "ctrlKey", "metaKey"
            ], jsHandlers: {}, jsAPI: null, _ihandlers: {}, onevents: {}, exportData: {} });
    }
    _defineHandler(name, body) {
        // adapted from SO to define a named handler
        let lines = ['return function ' + name + '(event, widget, context) {'];
        lines.push('\"use strict\";');
        lines.push(body);
        lines.push("}");
        return new Function(lines.join("\n"))();
    }
    _stringHash(str) {
        // just needed a simple one so: https://stackoverflow.com/questions/7616461/generate-a-hash-from-string-in-javascript
        var hash = 0, i, chr;
        if (str.length === 0)
            return hash;
        for (i = 0; i < str.length; i++) {
            chr = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + chr;
            hash |= 0; // Convert to 32bit integer
        }
        return hash;
    }
    _updateHandlers() {
        let handlers = this.get('jsHandlers');
        let debug = this.get('_debugPrint');
        let _ihandlers = this.get('_ihandlers');
        for (let h in handlers) {
            if (handlers.hasOwnProperty(h)) {
                let hash = this._stringHash(handlers[h]);
                if ((!_ihandlers.hasOwnProperty(h)) ||
                    (_ihandlers[h][0] !== hash)) {
                    if (debug) {
                        console.log('adding handler', h);
                    }
                    _ihandlers[h] = [hash, this._defineHandler(h, handlers[h])];
                }
            }
        }
    }
    _handle_comm_msg(msg) {
        const data = msg.content.data;
        let method = data.method;
        // if (this.get("_debugPrint")) {
        //     console.log("Message In:", data.method);
        // }
        if (method === "trigger") {
            this.trigger(data.content['handle'], data.content, msg.buffers);
            return Promise.resolve();
        }
        else if (method === "call") {
            this.callHandler(data.content['handle'], this.dummyEvent(data.content['handle'], { content: data.content, buffers: msg.buffers }));
            return Promise.resolve();
        }
        else {
            return super._handle_comm_msg(msg);
        }
    }
    dummyEvent(name, ops = {}) {
        return Object.assign({ target: this, type: name, stopPropagation: function () { } }, ops); // a hack only so we can use the same interface for custom events
    }
    callHandler(method, event) {
        let handlers = this.get('_ihandlers');
        let fn = null;
        if (handlers.hasOwnProperty(method)) {
            fn = handlers[method][1];
        }
        else {
            let api = this.get('jsAPI');
            if (api !== null) {
                handlers = api.get("_ihandlers");
                if (handlers.hasOwnProperty(method)) {
                    fn = handlers[method][1];
                }
            }
        }
        if (fn !== null) {
            fn.call(this, event, this, ActiveHTMLView.handlerContext);
        }
        else {
            throw new Error("couldn't find handler " + method);
        }
    }
}
exports.ActiveHTMLModel = ActiveHTMLModel;
ActiveHTMLModel.serializers = Object.assign(Object.assign({}, base_1.DOMWidgetModel.serializers), { 
    // Add any extra serializers here
    //@ts-ignore
    children: { deserialize: base_1.unpack_models }, 
    //@ts-ignore
    elementAttributes: { deserialize: base_1.unpack_models }, 
    //@ts-ignore
    exportData: { deserialize: base_1.unpack_models }, 
    //@ts-ignore
    jsAPI: { deserialize: base_1.unpack_models } });
ActiveHTMLModel.model_name = 'ActiveHTMLModel';
ActiveHTMLModel.model_module = version_1.MODULE_NAME;
ActiveHTMLModel.model_module_version = version_1.MODULE_VERSION;
ActiveHTMLModel.view_name = 'ActiveHTMLView'; // Set to null if no view
ActiveHTMLModel.view_module = version_1.MODULE_NAME; // Set to null if no view
ActiveHTMLModel.view_module_version = version_1.MODULE_VERSION;
class ActiveHTMLView extends base_1.DOMWidgetView {
    constructor() {
        super(...arguments);
        this._escapesMap = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
            '/': '&#x2F;',
            '`': '&#x60;',
            '=': '&#x3D;'
        };
    }
    // constructDict(listPair:any) {
    //     let res = {};
    //     let keys = listPair[0];
    //     let vals = listPair[1];
    //     for (let i = 0; i < keys.length; i++) {
    //         //@ts-ignore
    //         res[keys[i]] = vals[i];
    //     }
    //     return res;
    // }
    initialize(parameters) {
        super.initialize(parameters);
        //@ts-ignore
        this.children_views = new base_1.ViewList(this.add_child_model, null, this);
        this.listenTo(this.model, 'change:children', this.updateBody);
        this.listenTo(this.model, 'change:innerHTML', this.updateBody);
        this.listenTo(this.model, 'change:textContent', this.updateBody);
        this.listenTo(this.model, 'change:styleDict', this.updateStyles);
        this.listenTo(this.model, 'change:classList', this.updateClassList);
        this.listenTo(this.model, 'change:value', this.updateValue);
        this.listenTo(this.model, 'change:elementAttributes', this.updateAttributes);
        this.listenTo(this.model, 'change:eventPropertiesDict', this.updateEvents);
        this._currentEvents = {};
        this._currentOnHandlers = {};
        this._currentClasses = new Set();
        this._currentStyles = new Set();
        this._initted = false;
    }
    removeStyles() {
        let newStyles = this.model.get("styleDict");
        let current = this._currentStyles;
        return ActiveHTMLView._each(current, (prop) => {
            if (!newStyles.hasOwnProperty(prop)) {
                this.el.style.removeProperty(prop);
                this._currentStyles.delete(prop);
            }
        });
    }
    setLayout(layout, oldLayout) { } // null override
    setStyle(style, oldStyle) { } // null override
    setStyles() {
        let elementStyles = this.model.get("styleDict");
        let keys = Object.keys(elementStyles);
        if (keys.length === 0) {
            this._currentStyles.clear();
            this.el.removeAttribute('style');
            return ActiveHTMLView._defaultPromise();
        }
        else {
            if (this.model.get("_debugPrint")) {
                console.log(this.el, "Element Styles:", elementStyles);
            }
            return ActiveHTMLView._each(keys, (prop) => {
                if (elementStyles.hasOwnProperty(prop)) {
                    // console.log(">>>", prop, elementStyles[prop], typeof prop);
                    this.el.style.setProperty(prop, elementStyles[prop]);
                    // console.log("<<<", prop, this.el.style.getPropertyValue(prop));
                    this._currentStyles.add(prop);
                }
            });
        }
    }
    updateStyles() {
        return this.setStyles().then(() => this.removeStyles);
    }
    setClasses() {
        if (this.model.get("_debugPrint")) {
            console.log(this.el, "Element Classes:", this.model.get("classList"));
        }
        let classList = this.model.get("classList");
        return ActiveHTMLView._each(classList, (cls) => {
            this.el.classList.add(cls);
            this._currentClasses.add(cls);
        });
    }
    removeClasses() {
        if (this.model.get("_debugPrint")) {
            console.log(this.el, "Element Classes:", this.model.get("classList"));
        }
        let current = this._currentClasses;
        let classes = this.model.get("classList");
        return ActiveHTMLView._each(current, (cls) => {
            if (!classes.includes(cls)) {
                this.el.classList.remove(cls);
                this._currentClasses.delete(cls);
            }
        });
    }
    updateClassList() {
        return this.setClasses().then(() => this.removeClasses);
    }
    //manage body of element (borrowed from ipywidgets.Box)
    _createElement(tagName) {
        this.pWidget = new LayoutManagerWidget({ view: this });
        return this.pWidget.node;
    }
    _setElement(el) {
        if (this.el || el !== this.pWidget.node) {
            // Boxes don't allow setting the element beyond the initial creation.
            throw new Error('Cannot reset the DOM element.');
        }
        this.el = this.pWidget.node;
        this.$el = jquery_1.default(this.pWidget.node);
    }
    update_children() {
        if (this.children_views !== null) {
            return this.children_views.update(this.model.get('children')).then((views) => {
                // Notify all children that their sizes may have changed.
                views.forEach((view) => {
                    messaging_1.MessageLoop.postMessage(view.pWidget, widgets_1.Widget.ResizeMessage.UnknownSize);
                });
            });
        }
        else {
            return ActiveHTMLView._defaultPromise();
        }
    }
    add_child_model(model) {
        // we insert a dummy element so the order is preserved when we add
        // the rendered content later.
        let dummy = new widgets_1.Widget();
        //@ts-ignore
        this.pWidget.addWidget(dummy);
        return this.create_child_view(model).then((view) => {
            // replace the dummy widget with the new one.
            //@ts-ignore
            let i = algorithm_1.ArrayExt.firstIndexOf(this.pWidget.widgets, dummy);
            //@ts-ignore
            this.pWidget.insertWidget(i, view.pWidget);
            dummy.dispose();
            return view;
        }).catch(base_1.reject('Could not add child view to HTMLElement', true));
    }
    remove() {
        this.children_views = null;
        let onevents = this.model.get('onevents');
        if (onevents.hasOwnProperty('remove')) {
            let oninit = onevents['remove'];
            // if (Object.keys(oninit).length > 0) {
            this.handleEvent(this.dummyEvent('remove'), 'remove', oninit);
            // }
        }
        super.remove();
    }
    updateBody() {
        let children = this.model.get('children');
        let debug = this.model.get("_debugPrint");
        if (children.length > 0) {
            if (debug) {
                console.log(this.el, "Updating Children...");
            }
            return this.update_children();
        }
        else {
            let html = this.model.get("innerHTML");
            if (html.length > 0) {
                if (debug) {
                    console.log(this.el, "Updating HTML...");
                }
                this.updateInnerHTML();
                return ActiveHTMLView._defaultPromise();
            }
            else {
                let text = this.model.get("textContent");
                if (text.length > 0) {
                    if (debug) {
                        console.log(this.el, "Updating Text...");
                    }
                    this.updateTextContent();
                    return ActiveHTMLView._defaultPromise();
                }
                else {
                    if (debug) {
                        console.log(this.el, "Updating HTML...");
                    }
                    this.updateInnerHTML();
                    return ActiveHTMLView._defaultPromise();
                }
            }
        }
    }
    updateInnerHTML() {
        // let bodyType = this.model.get('_bodyType');
        // if (bodyType !== "html") {
        //   this.resetBody();
        // }
        let val = this.model.get("innerHTML");
        let cur = this.el.innerHTML;
        if (val !== cur) {
            this.el.innerHTML = val;
        }
        // if (bodyType !== "html") {
        //   this.model.set('_bodyType', "html");
        // }
    }
    updateTextContent() {
        // let bodyType = this.model.get('_bodyType');
        // if (bodyType !== "html") {
        //   this.resetBody(bodyType);
        // }
        let val = this.model.get("textContent");
        let cur = this.el.textContent;
        if (val !== cur) {
            this.el.textContent = val;
        }
        // if (bodyType !== "html") {
        //   this.model.set('_bodyType', "html");
        // }
    }
    getAttribute(attrName) {
        return this.model.get('elementAttributes')[attrName];
    }
    // Setting attributes (like id)
    updateAttribute(attrName) {
        let attrs = {};
        attrs[attrName] = this.model.get(attrName);
        return this._updateAttribute(attrName, attrs);
    }
    ;
    updateAttributeFromQuery(attrName, queryName) {
        let val = this.model.get(queryName);
        if (val === "") {
            this.el.removeAttribute(attrName);
        }
        else {
            this.el.setAttribute(attrName, val);
        }
    }
    _escapeHTML(html) {
        let parent = this;
        return String(html).replace(/[&<>"'`=\/]/g, function (s) {
            return parent._escapesMap[s];
        });
    }
    _modelHTMLSetter(prop, val) {
        let parent = this;
        function setHTML() {
            let ud = val.update();
            if (ud !== undefined) {
                return ud.then(() => {
                    let newHTML = val.el.outerHTML;
                    let oldHTML = parent.el.getAttribute(prop);
                    if (newHTML != oldHTML) {
                        parent.el.setAttribute(prop, newHTML);
                        return parent.notifyAttrUpdate(prop);
                    }
                    else {
                        return ActiveHTMLView._defaultPromise();
                    }
                });
            }
            else {
                let newHTML = val.el.outerHTML;
                let oldHTML = parent.el.getAttribute(prop);
                if (newHTML != oldHTML) {
                    parent.el.setAttribute(prop, newHTML);
                    return parent.notifyAttrUpdate(prop);
                }
                else {
                    return ActiveHTMLView._defaultPromise();
                }
            }
        }
        return setHTML;
    }
    _attachWidgetAsAttr(prop, val) {
        let setter = this._modelHTMLSetter(prop, val);
        val.model.on("change", setter, val);
        let r = val.render();
        if (r !== undefined) {
            return r.then(() => {
                if (val.hasOwnProperty('renderChildren')) {
                    //@ts-ignore
                    return val.renderChildren().then(setter);
                }
                else {
                    return setter();
                }
            });
        }
        else if (val.hasOwnProperty('renderChildren')) {
            //@ts-ignore
            return val.renderChildren().then(setter);
        }
        else {
            return setter();
        }
    }
    _updateAttribute(prop, attrs) {
        let val = attrs[prop];
        if (val === "") {
            if (this.el.hasAttribute(prop)) {
                this.el.removeAttribute(prop);
                return this.notifyAttrUpdate(prop);
            }
        }
        else if (typeof val === 'string') {
            let cur = this.el.getAttribute(prop);
            if (cur !== val) {
                this.el.setAttribute(prop, val);
                return this.notifyAttrUpdate(prop);
            }
        }
        else if (val instanceof base_1.WidgetView) {
            return this._attachWidgetAsAttr(prop, val);
        }
        else if (val instanceof base_1.WidgetModel) {
            return this.create_child_view(val).then((view) => {
                return this._attachWidgetAsAttr(prop, view);
            }).catch(base_1.reject('Could not add child view to HTMLElement', true));
        }
        else {
            this.el.setAttribute(prop, val.toString() + Object.keys(val).toString());
            return this.notifyAttrUpdate(prop);
        }
        return ActiveHTMLView._defaultPromise();
    }
    static _each(arr, fn) {
        return __awaiter(this, void 0, void 0, function* () {
            for (const item of arr)
                yield fn(item);
        });
    }
    static _defaultPromise(val = null) {
        return new Promise((resolve) => { resolve(val); });
    }
    updateAttributes() {
        let attrs = this.model.get('elementAttributes');
        let debug = this.model.get("_debugPrint");
        if (debug) {
            console.log(this.el, "Element Properties:", attrs);
        }
        return ActiveHTMLView._each(Object.keys(attrs), (prop) => this._updateAttribute(prop, attrs));
    }
    notifyAttrUpdate(prop) {
        let key = "view-change:" + prop;
        let onevents = this.model.get('onevents');
        if (onevents.hasOwnProperty('remove')) {
            if (this.model.get('_debugPrint')) {
                console.log('notifying attr change:', key);
            }
            let props = onevents[key];
            this.handleEvent(this.dummyEvent(key), key, props);
            // }
        }
        return ActiveHTMLView._defaultPromise();
    }
    updateValue() {
        let el = this.el;
        let debug = this.model.get("_debugPrint");
        if (el !== undefined) {
            let is_checkbox = el.getAttribute('type') === 'checkbox' || el.getAttribute('type') === 'radio';
            let multiple = el.getAttribute('multiple');
            if (is_checkbox) {
                let checked = el.checked;
                if (checked !== undefined) {
                    let newVal = this.model.get('value');
                    let checkVal = newVal.length > 0 && newVal != "false" && newVal != "0";
                    if (checkVal !== checked) {
                        if (debug) {
                            console.log(this.el, 'updating checked', checked, "->", checkVal);
                        }
                        el.checked = checkVal;
                    }
                }
            }
            else if (multiple) {
                let el = this.el;
                let opts = el.selectedOptions;
                if (opts !== undefined) {
                    let val = [];
                    for (let i = 0; i < opts.length; i++) {
                        let o = opts[i];
                        val.push(o.value);
                    }
                    let newValStr = this.model.get('value');
                    if (typeof newValStr === 'string') {
                        let testVal = val.join('&&');
                        if (newValStr !== testVal) {
                            if (debug) {
                                console.log(this.el, 'updating selection', testVal, "->", newValStr);
                            }
                            let splitVals = newValStr.split("&&");
                            for (let i = 0; i < el.options.length; i++) {
                                let o = el.options[i];
                                o.selected = (splitVals.indexOf(o.value) > -1);
                            }
                        }
                    }
                }
            }
            else {
                let val = el.value;
                if (val !== undefined) {
                    let newVal = this.model.get('value');
                    if (newVal !== val) {
                        if (debug) {
                            console.log(this.el, 'updating value', val, "->", newVal);
                        }
                        el.value = newVal;
                    }
                }
            }
        }
        return ActiveHTMLView._defaultPromise();
    }
    _registerEvent(key, listeners) {
        if (!this._currentEvents.hasOwnProperty(key)) {
            this._currentEvents[key] = [
                listeners[key],
                this.constructEventListener(key, listeners[key])
            ];
            this.el.addEventListener(key, this._currentEvents[key][1]);
        }
        else if (this._currentEvents[key][0] !== listeners[key]) {
            this.el.removeEventListener(key, this._currentEvents[key][1]);
            this._currentEvents[key] = [
                listeners[key],
                this.constructEventListener(key, listeners[key])
            ];
            this.el.addEventListener(key, this._currentEvents[key][1]);
        }
    }
    setEvents() {
        let listeners = this.model.get('eventPropertiesDict');
        let debug = this.model.get("_debugPrint");
        if (debug) {
            console.log(this.el, "Adding Events:", listeners);
        }
        return ActiveHTMLView._each(Object.keys(listeners), (key) => {
            if (listeners.hasOwnProperty(key)) {
                this._registerEvent(key, listeners);
            }
        });
    }
    removeEvents() {
        let newListeners = this.model.get('eventPropertiesDict');
        let current = this._currentEvents;
        let debug = this.model.get("_debugPrint");
        return ActiveHTMLView._each(Object.keys(current), (prop) => {
            if (current.hasOwnProperty(prop)) {
                if (!newListeners.hasOwnProperty(prop)) {
                    if (debug) {
                        console.log(this.el, "Removing Event:", prop);
                    }
                    this.el.removeEventListener(prop, this._currentEvents[prop][1]);
                    this._currentEvents.delete(prop);
                }
            }
        });
    }
    updateEvents() {
        return this.setEvents().then(() => this.removeEvents);
    }
    _registerOnHandler(key, listeners) {
        if (!this._currentOnHandlers.hasOwnProperty(key)) {
            this._currentOnHandlers[key] = [
                listeners[key],
                this.constructOnHandler(key, listeners[key])
            ];
            this.model.on(key, this._currentOnHandlers[key][1], this);
        }
        else if (this._currentOnHandlers[key][0] !== listeners[key]) {
            this.model.off(key, this._currentOnHandlers[key][1], this);
            this._currentOnHandlers[key] = [
                listeners[key],
                this.constructOnHandler(key, listeners[key])
            ];
            this.model.on(key, this._currentOnHandlers[key][1], this);
        }
    }
    setOnHandlers() {
        let listeners = this.model.get('onevents');
        let debug = this.model.get("_debugPrint");
        if (debug) {
            console.log(this.el, "Adding On Handlers:", listeners);
        }
        return ActiveHTMLView._each(Object.keys(listeners), (key) => {
            if (listeners.hasOwnProperty(key)) {
                return this._registerOnHandler(key, listeners);
            }
        });
    }
    removeOnHandlers() {
        let newListeners = this.model.get('onevents');
        let current = this._currentOnHandlers;
        let debug = this.model.get("_debugPrint");
        return ActiveHTMLView._each(Object.keys(current), (prop) => {
            if (current.hasOwnProperty(prop)) {
                if (!newListeners.hasOwnProperty(prop)) {
                    if (debug) {
                        console.log(this.el, "Removing On Handler:", prop);
                    }
                    this.model.off(prop, current[prop][1], this);
                    current.delete(prop);
                }
            }
        });
    }
    updateOnHandlers() {
        return this.setOnHandlers().then(() => this.removeOnHandlers);
    }
    _calloninit() {
        if (!this._initted) {
            let onevents = this.model.get('onevents');
            if (onevents.hasOwnProperty('initialize')) {
                let oninit = onevents['initialize'];
                if (Object.keys(oninit).length > 0) {
                    this.handleEvent(this.dummyEvent('initialize'), 'initialize', oninit);
                }
            }
        }
        this._initted = true;
    }
    render() {
        let r = super.render();
        if (r !== undefined) {
            return r.then((v) => {
                this.el.classList.remove('lm-Widget', 'p-Widget');
                return this.update().then(() => this._calloninit);
            });
        }
        else {
            this.el.classList.remove('lm-Widget', 'p-Widget');
            return this.update().then((v) => { this._calloninit(); return v; });
        }
    }
    renderChildren() {
        if (this.children_views !== null) {
            return this.children_views.update([]).then((views) => ActiveHTMLView._each(views, (v) => {
                if (v.hasOwnProperty('renderChildren')) {
                    //@ts-ignore
                    return v.renderChildren();
                }
                else {
                    v.render();
                }
            }));
        }
        else {
            return ActiveHTMLView._defaultPromise();
        }
    }
    update() {
        return this.updateAttribute('id').then(() => this.updateClassList().then(() => this.setStyles().then(() => this.updateBody().then(() => this.updateAttributes().then(() => this.setEvents().then(() => this.setOnHandlers().then(() => this.updateValue().then(() => { return this; }))))))));
    }
    // @ts-ignore
    get tagName() {
        // We can't make this an attribute with a default value
        // since it would be set after it is needed in the
        // constructor.
        return this.model.get('tagName');
    }
    // Adapted from the "TextView" from the core package
    events() {
        let events = {};
        if (this.model.get('trackInput')) {
            // let tagName = this.model.get('tagName');
            let key = 'keydown'; // '.concat(tagName);
            //@ts-ignore
            events[key] = 'handleKeyDown';
            key = 'keypress'; // '.concat(tagName);
            //@ts-ignore
            events[key] = 'handleKeypress';
            key = 'input'; // '.concat(tagName);
            //@ts-ignore
            events[key] = 'handleChanging';
            key = 'change'; // '.concat(tagName);
            //@ts-ignore
            events[key] = 'handleChanged';
        }
        return events;
    }
    handleKeyDown(e) {
        e.stopPropagation();
    }
    handleKeypress(e) {
        e.stopPropagation();
    }
    handleChanging(e) {
        if (this.model.get('continuousUpdate')) {
            this.handleChanged(e);
        }
    }
    handleChanged(e) {
        let target = e.target;
        let el = this.el;
        let is_checkbox = el.getAttribute('type') === 'checkbox' || el.getAttribute('type') === 'radio';
        let multiple = el.getAttribute('multiple');
        if (is_checkbox) {
            let checked = target.checked;
            if (checked !== undefined) {
                this.model.set('value', checked ? "true" : "false", { updated_view: this });
                this.touch();
            }
        }
        else if (multiple) {
            let el = this.el;
            let opts = el.selectedOptions;
            if (opts !== undefined) {
                let val = [];
                for (let i = 0; i < opts.length; i++) {
                    let o = opts[i];
                    val.push(o.value);
                }
                let newVal = val.join('&&');
                this.model.set('value', newVal, { updated_view: this });
                this.touch();
            }
        }
        else {
            let val = target.value;
            if (val !== undefined) {
                this.model.set('value', val, { updated_view: this });
                this.touch();
            }
        }
    }
    setData(key, value) {
        let data = this.model.get('exportData');
        data[key] = value;
        this.model.set('exportData', {}, { updated_view: this }); // force a state change
        this.model.set('exportData', data, { updated_view: this });
        this.touch();
    }
    getData(key, value) {
        let data = this.model.get('exportData');
        return data[key];
    }
    handleEvent(e, eventName, propData) {
        let props;
        let method = "";
        let send = true;
        if (Array.isArray(propData)) {
            props = propData;
        }
        else if (propData === undefined || propData === null) {
            props = this.model.get('defaultEventProperties');
        }
        else if (typeof propData === 'string') {
            method = propData;
            props = [];
        }
        else {
            method = propData['method'];
            if (method === undefined || method === null) {
                method = "";
            }
            else {
                send = propData.hasOwnProperty('notify') && propData['notify'] === true;
            }
            if (propData.hasOwnProperty('fields')) {
                props = propData['fields'];
            }
            else {
                props = this.model.get('defaultEventProperties');
            }
            let prop = propData['propagate'] !== true;
            if (prop) {
                e.stopPropagation();
            }
        }
        let debug = this.model.get('_debugPrint');
        if (debug) {
            console.log(this.el, "Handling event:", eventName, propData);
            if (method !== "") {
                console.log(this.el, "calling handler", method);
            }
        }
        // console.log("|", eventName, props);
        if (method !== "") {
            this.callHandler(method, e);
        }
        if (send) {
            this.sendEventMessage(e, this.constructEventMessage(e, props, eventName));
        }
    }
    callHandler(method, event) {
        let handlers = this.model.get('_ihandlers');
        let fn = null;
        if (handlers.hasOwnProperty(method)) {
            fn = handlers[method][1];
        }
        else {
            let api = this.model.get('jsAPI');
            if (api !== null) {
                handlers = api.get("_ihandlers");
                if (handlers.hasOwnProperty(method)) {
                    fn = handlers[method][1];
                }
            }
        }
        if (fn !== null) {
            fn.call(this, event, this, ActiveHTMLView.handlerContext);
        }
        else {
            throw new Error("couldn't find handler " + method);
        }
    }
    dummyEvent(name, ops = {}) {
        return Object.assign({ target: this.el, type: name, stopPropagation: function () { } }, ops); // a hack only so we can use the same interface for custom events
    }
    constructEventListener(eventName, propData) {
        let parent = this;
        return function (e) {
            parent.handleEvent(e, eventName, propData);
        };
    }
    constructOnHandler(eventName, propData) {
        let listener = this.constructEventListener(eventName, propData);
        let event = this.dummyEvent(eventName);
        return function (msg) {
            return listener(Object.assign(Object.assign({}, msg), event));
        };
    }
    constructEventMessage(e, props, eventName) {
        if (props === undefined || props === null) {
            props = this.model.get('defaultEventProperties');
        }
        if (props === undefined) {
            props = ['target'];
        }
        let eventMessage = {};
        if (eventName !== undefined) {
            eventMessage['eventName'] = eventName;
        }
        for (let p of props) {
            // @ts-ignore
            let val = e[p];
            if (p === "target") {
                val = {};
                let t = e.target;
                val['tag'] = t.tagName;
                val['innerHTML'] = t.innerHTML;
                for (let p of t.getAttributeNames()) {
                    val[p] = t.getAttribute(p);
                }
            }
            eventMessage[p] = val;
        }
        return eventMessage;
    }
    sendEventMessage(e, message) {
        if (message === undefined) {
            message = this.constructEventMessage(e);
        }
        let debug = this.model.get('_debugPrint');
        if (debug) {
            console.log(this.el, "Sending message:", message);
        }
        this.send(message);
    }
}
exports.ActiveHTMLView = ActiveHTMLView;
ActiveHTMLView.handlerContext = {
    'bootstrap': bootstrap,
    "$": jquery_1.default,
    "jquery": jquery,
};
//# sourceMappingURL=widget.js.map

/***/ }),

/***/ "./package.json":
/*!**********************!*\
  !*** ./package.json ***!
  \**********************/
/***/ ((module) => {

module.exports = JSON.parse('{"name":"ActiveHTMLWidget","version":"0.1.0","description":"A Custom Jupyter Widget Library","keywords":["jupyter","jupyterlab","jupyterlab-extension","widgets"],"files":["lib/**/*.js","dist/*.js","css/*.css"],"homepage":"https://github.com//ActiveHTMLWidget","bugs":{"url":"https://github.com//ActiveHTMLWidget/issues"},"license":"MIT","author":{"name":"b3m2a1","email":"b3m2a1@gmail.com"},"main":"lib/index.js","types":"./lib/index.d.ts","style":"css/index.css","sideEffects":["css/*.css"],"repository":{"type":"git","url":"https://github.com//ActiveHTMLWidget"},"scripts":{"build":"yarn run build:lib && yarn run build:nbextension && yarn run build:labextension:dev","build:prod":"yarn run build:lib && yarn run build:nbextension && yarn run build:labextension","build:labextension":"jupyter labextension build .","build:labextension:dev":"jupyter labextension build --development True .","build:lib":"tsc","build:nbextension":"webpack","clean":"yarn run clean:lib && yarn run clean:nbextension && yarn run clean:labextension","clean:lib":"rimraf lib","clean:labextension":"rimraf ActiveHTMLWidget/labextension","clean:nbextension":"rimraf ActiveHTMLWidget/nbextension/static/index.js","lint":"eslint . --ext .ts,.tsx --fix","lint:check":"eslint . --ext .ts,.tsx","prepack":"yarn run build:lib","test":"jest","watch":"npm-run-all -p watch:*","watch:lib":"tsc -w","watch:nbextension":"webpack --watch --mode=development","watch:labextension":"jupyter labextension watch ."},"dependencies":{"@jupyter-widgets/base":"^1.1.10 || ^2.0.0 || ^3.0.0 || ^4.0.0","bootstrap":"^5.1.3"},"devDependencies":{"@babel/core":"^7.5.0","@babel/preset-env":"^7.5.0","@jupyterlab/builder":"^3.0.0","@phosphor/application":"^1.6.0","@phosphor/widgets":"^1.6.0","@types/bootstrap":"^5.1.11","@types/jest":"^26.0.0","@types/webpack-env":"^1.13.6","@typescript-eslint/eslint-plugin":"^3.6.0","@typescript-eslint/parser":"^3.6.0","acorn":"^7.2.0","css-loader":"^3.2.0","eslint":"^7.4.0","eslint-config-prettier":"^6.11.0","eslint-plugin-prettier":"^3.1.4","fs-extra":"^7.0.0","identity-obj-proxy":"^3.0.0","jest":"^26.0.0","mkdirp":"^0.5.1","npm-run-all":"^4.1.3","prettier":"^2.0.5","rimraf":"^2.6.2","sass":"^1.51.0","source-map-loader":"^1.1.3","style-loader":"^1.0.0","ts-jest":"^26.0.0","ts-loader":"^8.0.0","typescript":"~4.1.3","webpack":"^5.0.0","webpack-cli":"^4.0.0"},"jupyterlab":{"extension":"lib/plugin","outputDir":"ActiveHTMLWidget/labextension/","sharedPackages":{"@jupyter-widgets/base":{"bundled":false,"singleton":true}}}}');

/***/ })

}]);
//# sourceMappingURL=lib_widget_js.950f05f0927eac55615b.js.map